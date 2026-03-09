import bpy
import os

def setup_render(width, height):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"

def render_image(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)