from tkinter import N
from scene import Scene
import taichi as ti
from taichi.math import *
import math

scene = Scene(exposure=1.)
scene.set_floor(-1., (0.9, 0.9, 0.9))
#scene.set_ambient_light((0.1,0.15,0.15))
scene.set_directional_light((1, 1, 2), 0.02, (2., 2., 1.5))
scene.set_background_color((0.2, 0.25, 0.3)) #((0.3, 0.4, 0.6))

#SDF copied from one of the examples.
@ti.static
@ti.func
def make_nested(f):
    f = f * 40
    i = int(f)
    if f < 0:
        if i % 2 == 1:
            f -= ti.floor(f)
        else:
            f = ti.floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f

@ti.func
def My_SDF_func(o):
    wall = min(o[1] + 0.1, o[2] + 0.4)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36

    q = ti.abs(o - ti.Vector([0.8, 0.3, 0])) - ti.Vector([0.3, 0.3, 0.3])
    box = ti.Vector([max(0, q[0]), max(0, q[1]),
                     max(0, q[2])]).norm() + min(q.max(), 0)

    O = o - ti.Vector([-0.8, 0.3, 0])
    d = ti.Vector([ti.Vector([O[0], O[2]]).norm() - 0.3, abs(O[1]) - 0.3])
    cylinder = min(d.max(), 0.0) + ti.Vector([max(0, d[0]),
                                              max(0, d[1])]).norm()

    geometry = make_nested(min(sphere, box, cylinder))
    geometry = max(geometry, -(0.32 - (o[1] * 0.6 + o[2] * 0.8)))
    return min(wall, geometry)

@ti.func
def My_SDF_col(o, n):
    return ti.Vector([0.75, 0.4, 0.5]) + 0.3 * n

#the SDF and color functions are called from the renderer so let's tell it which functions to use.
scene.set_sdf_func(My_SDF_func)
scene.set_sdf_col(My_SDF_col)

#help preserving your GC: set it to something like 1000 if you want a high quality / lower noise results
scene.maxSamples = 100

#Where the magic is
scene.finish()
