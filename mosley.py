from tkinter import N
from scene import Scene
import taichi as ti
from taichi.math import *
import math

scene = Scene(exposure=1.)
scene.set_floor(-1.1, (0.9, 0.9, 0.9))
#scene.set_ambient_light((0.1,0.15,0.15))
scene.set_directional_light((1, 1, 1), 0.005, (2., 2., 1.5))
scene.set_background_color((0.2, 0.25, 0.3)) #((0.3, 0.4, 0.6))

#To be replaced
@ti.func
def My_SDF_col(o, n):
    return ti.Vector([0.75, 0.4, 0.5]) + 0.3 * n

@ti.func
def boxDE(p, siz):
    '''
    Good old box. Correct SDF à la IQ (err... more or less.)
    '''
    z = ti.abs(p) #ti.Vector([ti.abs(p[0]), ti.abs(p[1]), ti.abs(p[2])])
    if z[0] < z[1] : z[0], z[1] = z[1], z[0]
    if z[0] < z[2] : z[0], z[2] = z[2], z[0] 
    if z[1] < z[2] : z[1], z[2] = z[2], z[1]
    z -= ti.Vector([siz, siz, siz])
    de = z[0]
    if z[2] > 0 :
        de = z.norm()
    else: 
        if z[1] > 0 : 
            de =  ti.Vector([z[0], z[1]]).norm()
    return de

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Taken from math_utils.py
    """
    axis = axis.normalized()
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return ti.Matrix([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], ti.f32)

#I you are wondering why the ti.field... stuff, it's because I was (still am) struggling with interactive modification using GUI
MAX_ITER = 14
MaxIter = ti.field(dtype=ti.f32, shape=())
MaxIter[None] = MAX_ITER
Offset = ti.Vector([1,1,0])
Scale = ti.field(dtype=ti.f32, shape=())
Scale[None] = 3.

RotAngle = ti.field(dtype=ti.f32, shape=())
RotAngle[None] = 0
Rot = ti.Matrix([[1,0,0],[0,1,0],[0,0,1]], ti.f32)
Rot = rotation_matrix(ti.Vector([0,1,0]), RotAngle[None] * math.pi / 180)

@ti.func
def Mosley(p):
    '''
    Mosley fractal: A lot of Koch snowflakes hiding there.
    '''
    r2 = p.norm()
    dd = 1.
    for i in range(MaxIter[None]): #(MAX_ITER) :
        if p.dot(p) > 100. : break #if p.norm() > 10. : break
        #fold
        p = ti.abs(p)                               #fold along x,y and z axes
        if p[0] < p[1] : p[0], p[1] = p[1], p[0]    #fold along diagonals
        if p[1] < p[2] : p[1], p[2] = p[2], p[1] 
        if p[0] < p[1] : p[0], p[1] = p[1], p[0]
        #p[0] = ti.abs(p[0] - 1./3. * Offset[0]) + 1./3. * Offset[0]
        p[1] = ti.abs(p[1] - 1./3. * Offset[1]) + 1./3. * Offset[1]
        #p[2] = ti.abs(p[2] - 1./3. * Offset[2]) + 1./3. * Offset[2]

        p = p * Scale[None] - Offset * (Scale[None] -1.)
        dd*= Scale[None]
        p = Rot @ p
    #using that boxDE gives nicer results at low iteration count
    return (boxDE(p, 1.)-0.) / dd

#the SDF and color functions are called from the renderer so let's tell it which functions to use.
scene.set_sdf_func(Mosley)
scene.set_sdf_col(My_SDF_col)

#help preserving your GC: set it to something like 1000 if you want a high quality / lower noise results
scene.maxSamples = 100

#trying GUI
##Works but much slower so it is deactivated. See also scene.finish() in scene.py
###BTW! rotation matrix doesn't work.
def myGUI(win):
    Modified = False
    win.GUI.begin("Mosley fractal", 0.05, 0.05, 0.3, 0.2)
    win.GUI.text("Simple IFS and kaleidoscopic fractal")
    win.GUI.button("bla")
    RotAngle[None] = win.GUI.slider_float("Angle",RotAngle[None],0,360)
    
    amxitr = ti.field(dtype=ti.f32, shape=())
    amxitr[None] = MaxIter[None]
    MaxIter[None] = ti.round( win.GUI.slider_float("Max iterations",amxitr[None],0,20) )
    if amxitr[None]!=MaxIter[None]: Modified = True
    
    ascl = ti.field(dtype=ti.f32, shape=())
    ascl[None] = Scale[None]
    Scale[None] = win.GUI.slider_float("Scale",ascl[None],1.1,3)
    if ascl[None]!=Scale[None]: Modified = True

    win.GUI.end()

    Rot = rotation_matrix(ti.Vector([0,1,0]), RotAngle[None] * math.pi / 180)

    return False #Modified

#tell scene which GUI callback to use
#scene.setGUICB(myGUI)

scene.finish()
