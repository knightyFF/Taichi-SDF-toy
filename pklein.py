#from tkinter import N
from scene import Scene
import taichi as ti
from taichi.math import *


scene = Scene(exposure=1.)
scene.set_floor(-10.1, (0.9, 0.9, 0.9))
#scene.set_ambient_light((0.1,0.15,0.15)) #buggy bug
scene.set_directional_light((1, 1, 2), 0.02, (2.16, 2.14, 1.8))
scene.set_background_color((0.1, 0.1, 0.15)) #((0.3, 0.4, 0.6))

@ti.func
def My_SDF_col(o, n):
    return vec3(0.5) + 0.1 * n

MAX_ITER = 10
MaxIter = ti.field(dtype=ti.f32, shape=())
MaxIter[None] = MAX_ITER

CSize  = vec3([0.9009688679, 1.1, 0.7071])
JuliaC = vec3(0)
Offset = vec3(0, 0.05, 0)
Size   = 1
BshapeSize = 2


@ti.func
def bShape(p, e):
    p -= Offset
    rxy = length(p.xz) - BshapeSize
    d0  = (length(p.xz) * ti.abs(p.y) - e) / ti.sqrt(p.dot(p) + abs(e))
    return ti.max(rxy, d0)

@ti.func
def pKlein(p):
    '''
    This is just a scale 1 Mandelbox julia with a geometric orbit trap
    '''
    DE0 = p[1] - 1
    DEfactor = 2.
    for i in range(MaxIter[None]):
        #Box fold
        p = 2 * clamp(p, -CSize, CSize) - p
        #Sphere fold
        r2 = dot(p, p)
        k  = max(Size / r2, 1.)
        p *= k
        DEfactor *= k
        #julia seed
        p += JuliaC
    #call basic shape A.K.A. the geometric orbit trap
    DEfractal = bShape(p, 0.1) / ti.abs(DEfactor)
    return max(DE0, DEfractal)

scene.renderer.ray_march_sdf_steps = 200
scene.set_sdf_func(pKlein)
scene.set_sdf_col(My_SDF_col)

#Kn: helps preserving your GC
scene.maxSamples = 100

#Kn: trying GUI
#Makes things a lot slower so not used :-/
def myGUI(win):
    Modified = False

    amxitr = ti.field(dtype=ti.f32, shape=())
    amxitr[None] = MaxIter[None]
    MaxIter[None] = ti.round(win.GUI.slider_float("Max iterations", amxitr[None],0,20) )
    if amxitr[None]!=MaxIter[None]: Modified = True

    win.GUI.end()

    return False #Modified

#scene.setGUICB(myGUI)

scene.finish()
