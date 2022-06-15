import taichi as ti

from math_utils import (eps, inf, out_dir, ray_aabb_intersection)

@ti.func
def default_SDF(o):
    wall = min(o[1] + 0.2, o[2] + 0.)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.3
    return ti.max(-wall,sphere)

@ti.func
def default_SDF_color(o, n):
    return ti.Vector([0.1, 0.5, 0.3])

MAX_RAY_DEPTH = 2
use_directional_light = True

DIS_LIMIT = 20


@ti.data_oriented
class Renderer:
    def __init__(self, image_res, up, exposure=3):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.fov = ti.field(dtype=ti.f32, shape=())
        
        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ambient_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        
        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.23)

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)
        
        self.sdf = default_SDF
        self.sdf_color = default_SDF_color
        self.ray_march_sdf_steps = 100

    def set_directional_light(self, direction, light_direction_noise,
                              light_color):
        direction_norm = (direction[0]**2 + direction[1]**2 +
                          direction[2]**2)**0.5
        self.light_direction[None] = (direction[0] / direction_norm,
                                      direction[1] / direction_norm,
                                      direction[2] / direction_norm)
        self.light_direction_noise[None] = light_direction_noise
        self.light_color[None] = light_color

    def set_ambient_light(self, light_color):
        self.ambient_color[None] = light_color

    @ti.func
    def ray_march_floor(self, p, d):
        '''
        Actually just finds the intersection of the ray and the horizontal plane
        '''
        dist = inf
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]
        return dist
    
    @ti.func
    def ray_march_sdf(self,p, d):
        '''
        Sphere tracing the scene represented in self.sdf().
        self.sdf() is provided by the user.
        '''
        j = 0
        dist = 0.0
        while j < self.ray_march_sdf_steps and self.sdf(p + dist * d) > 1e-4 * dist and dist < DIS_LIMIT : #inf:
            dist += self.sdf(p + dist * d)
            j += 1
        return min(inf, dist)

    @ti.func
    def get_floor_normal(self, p):
        return ti.Vector([0.0, 1.0, 0.0])  # up of course

    @ti.func
    def get_floor_color(self, p):
        return self.floor_color[None]

    @ti.func
    def get_sdf_normal(self, p):
        '''
        Computes the sdf's normal at p
        '''
        d = 1e-4
        n = ti.Vector([0.0, 0.0, 0.0])
        sdf_center = self.sdf(p)
        for i in ti.static(range(3)):
            inc = p
            inc[i] += d
            n[i] = (1 / d) * (self.sdf(inc) - sdf_center)
        return n.normalized()

    @ti.func
    def get_sdf_color(self, p, n):
        return self.sdf_color(p,n) # self.floor_color[None]

    @ti.func
    def next_hit(self, pos, d, t):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0

        #sdf
        ray_march_dist = self.ray_march_sdf(pos, d)
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.get_sdf_normal(pos + d * closest)
            c = self.get_sdf_color(pos + d * closest, normal)

        #floor
        ray_march_dist = self.ray_march_floor(pos, d)
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.get_floor_normal(pos + d * closest)
            c = self.get_floor_color(pos + d * closest)

        return closest, normal, c, hit_light

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

    @ti.func
    def get_cast_dir(self, u, v):
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    @ti.kernel
    def render(self):
        ti.loop_config(block_dim=256)
        for u, v in self.color_buffer:
            d = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]
            t = 0.0

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            c = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_light = 0
            hit_background = 0

            # Tracing begin
            for bounce in range(MAX_RAY_DEPTH):
                depth += 1
                closest, normal, c, hit_light = self.next_hit(pos, d, t)
                hit_pos = pos + closest * d
                if not hit_light and normal.norm() != 0 and closest < 1e8:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * self.light_direction_noise[None]
                        light_dir = (self.light_direction[None] +
                                     dir_noise).normalized()
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_ = self.next_hit(
                                pos, light_dir, t)
                            if dist > DIS_LIMIT:
                                # far enough to hit directional light
                                contrib += throughput * \
                                    self.light_color[None] * dot
                else:  # hit background or light voxel, terminate tracing
                    hit_background = 1
                    #Kn: Add light from "Sky"
                    #contrib += throughput * self.background_color[None]
                    break

                # Russian roulette
                max_c = throughput.max()
                if ti.random() > max_c:
                    throughput = [0, 0, 0]
                    break
                else:
                    throughput /= max_c
            # Tracing end

            if hit_light:
                contrib += throughput * c
            #else:
                #if depth == 1 and hit_background:
            if hit_background:
                # Direct hit to background
                contrib += throughput * self.background_color[None]
                #else:
            #Kn: "Ambient" light
            contrib += throughput * self.ambient_color[None]
    
            self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = ti.sqrt(
                    self.color_buffer[i, j][c] * darken * self.exposure /
                    samples)

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self):
        self.render()
        self.current_spp += 1

    def fetch_image(self):
        self._render_to_image(self.current_spp)
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c):
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r

