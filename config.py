import os
import os.path as osp
import pathlib

DEBUG = True

FOV_X = 63.4149

PATH_APP = str(pathlib.Path(__file__).parent.resolve())

PATH_MITSUBA = "..\\..\\Mitsuba\\mitsuba.exe"  # note that if use "start xxx.exe", the progress will not be blocked
PATH_IMAGE_IN = "data/debug/scene17.png"
PATH_IMAGE_OUT = "data/debug/scene01_out_web.png"

PATH_DATA_NORMAL = "data/debug/scene01/scene01_normal.png"
PATH_DATA_ALBEDO = "data/debug/scene01/scene01_albedo.png"
PATH_DATA_ROUGH = "data/debug/scene01/scene01_rough.png"
PATH_DATA_ENVMAP = "data/debug/scene01/scene01_probe00.exr"
PATH_DATA_OBJECT = "data/debug/bunny.obj"
PATH_DATA_OBJECT_BUNNY = "data/debug/bunny.obj"

PATH_OUT = "data/debug/scene01_out/"
PATH_OUT_PLANE = osp.join(PATH_OUT, "plane.obj")
PATH_OUT_XML = "data/debug/scene01_out/"

