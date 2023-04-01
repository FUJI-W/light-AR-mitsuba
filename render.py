import os
import subprocess
import os.path as osp
import pandas as pd
from utils import *
from PIL import Image
import cv2
import math
from util_xml import *


def getMatrixReverse(h, w, fovX=config.FOV_X):
    """
    get the reverse matrix to transfer coordinates from image space to world space

    Args:
        h: height of the image
        w: width of the image
        fovX: field of view (X) of the image

    Returns:
        m_reverse: the reverse matrix - (x,y,z)^T = m_reverse (x',y',z')^T
    """
    tan_fovX_2 = math.tan(math.radians(fovX / 2.0))
    tan_fovY_2 = tan_fovX_2 / (w / h)
    m_persp = np.asarray([[-1 / tan_fovX_2, 0, 0, 0],
                          [0, -1 / tan_fovY_2, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    m_persp_I = np.linalg.inv(m_persp)  # inverse matrix of m_persp
    m_viewport = np.asarray([[w / 2, 0, 0, w / 2],
                             [0, h / 2, 0, h / 2],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    m_viewport_I = np.linalg.inv(m_viewport)  # inverse matrix of m_viewport
    m_reverse = m_persp_I @ m_viewport_I

    return m_reverse


def generatePlaneMask(h, w, quad):
    """
    generate mask of picked plane

    Args:
        h: height of image (also the mask)
        w: width of image (also the mask)
        quad: 4 points of quadrangle - [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (np.array)

    Returns:
        mask: 2D mask of picked plane
    """
    grid_x, grid_y = np.meshgrid(range(1, w + 1), range(1, h + 1))
    mask = np.ones((h, w))
    for i in range(4):
        vec_quad = quad[(i + 1) % 4] - quad[i]  # shape: (2)
        mtx_point = [grid_x - quad[i][0], grid_y - quad[i][1]]  # shape: (2, h, w)
        cross_product = mtx_point[0] * vec_quad[1] - mtx_point[1] * vec_quad[0]  # shape: (h, w)
        mask = mask * (cross_product > 0)  # if cross_product > 0, point is in the quadrangle

    # cv2.imshow("mask", mask) if config.DEBUG else None
    # cv2.waitKey(0) if config.DEBUG else None
    return mask


def generatePlane3D(h, w, quad, mask, path_normal, path_plane, plane_vn=None):
    """
    generate 3D points of picked plane

    Args:
        h: height of image (also the mask)
        w: width of image (also the mask)
        quad: 4 points of quadrangle - [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (np.array)
        mask: 2D mask of picked plane
        path_normal: path to normal of the image
        path_plane:  path to output the plane.obj
        plane_vn: if plane_vn is to be calculated

    Returns:
        vn: normal of plane - np.array(3,1)
        v: 4 vertices of plane - np.array(4,3)
        vt: texture coordinates of each vertex - np.array(4,2)
    """
    vn = []  # shape: (3,1)
    if plane_vn is None:
        if path_normal.endswith(".npy"):
            normal = np.load(path_normal)
        else:
            normal = np.asarray(Image.open(path_normal))
            normal = normal / 127.5 - 1.0
        for i in range(3):
            ni = normal[:, :, i]
            ni_mean = np.mean(ni[mask == 1])
            vn.append([ni_mean])
        vn = np.asarray(vn)
        vn = vn / (np.sqrt(np.sum(vn * vn)))  # normalize
        vn = vn * np.asarray([[-1], [1], [-1]])  # right hand -> left hand
    else:
        vn = plane_vn
    print("\nPlane's vn:\n", vn) if config.DEBUG else None

    v = []  # shape: (4,3)
    m_reverse = getMatrixReverse(h, w, fovX=config.FOV_X)
    for q in quad:
        coord = np.asarray([[q[0]], [q[1]], [1], [1]])
        coord = m_reverse @ coord
        v.append(coord[:3].T.squeeze())
    v = np.asarray(v)
    for i in range(1, 4):
        # vec_edge = v[i] - v[0]  # vec_edge @ vn should be zero, as vn is plane's normal
        # v[i][2] = -(vec_edge @ vn) / vn[2] + v[0][2]  # follow the up rule, get z of vi
        ratio = (v[0] @ vn) / (v[i] @ vn)
        assert ratio > 0  # vi·vn should equal v0·vn and share same direction, as vn is plane's normal
        v[i] = ratio * v[i]
        assert math.isclose((v[i] - v[0]) @ vn, 0, abs_tol=0.001)  # vec_edge @ vn should be zero, as vn is plane's normal
    print("\nPlane's v:\n", v, "\n") if config.DEBUG else None

    vt = []  # shape: (4,2)
    m_texture = np.asarray([[w, 0, 0],
                            [0, -h, h],
                            [0, 0, 1]])
    m_texture_I = np.linalg.inv(m_texture)  # inverse matrix of m_texture
    for q in quad:
        coord = np.asarray([[q[0]], [q[1]], [1]])
        coord = m_texture_I @ coord
        vt.append(coord[:2].T.squeeze())
    vt = np.asarray(vt)
    print("\nPlane's vt:\n", vt, "\n") if config.DEBUG else None

    if path_plane != "":
        output2obj(path=path_plane, vn=vn, v=v, vt=vt, mode="quad")
        print("\nOutput the plane.obj to", path_plane) if config.DEBUG else None

    return vn, v, vt


def generateObject3D(h, w, point, quad, quad_vn, quad_v):
    """
    generate the 3D information of the insert position

    Args:
        h: height of image
        w: width of image
        point: position to insert object - [x,y] (np.array)
        quad: picked plane - [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (np.array)
        quad_vn: normal of plane - np.array(3,1)
        quad_v: 4 vertices of plane - np.array(4,3)

    Returns:
        obj_v: 3D coordinates of insert position
        obj_v_img: [still unknown]
    """
    assert isPointInQuadrangle(point, quad)

    coord = np.asarray([[point[0]], [point[1]], [1], [1]])
    m_reverse = getMatrixReverse(h, w, fovX=config.FOV_X)
    obj_v = (m_reverse @ coord)[:3].T.squeeze()
    obj_v[2] = -((obj_v - quad_v[0]) @ quad_vn) / quad_vn[2] + quad_v[0][2]
    assert math.isclose((obj_v - quad_v[0]) @ quad_vn, 0, abs_tol=0.001)
    print("\nObject's v:\n", obj_v) if config.DEBUG else None

    obj_v_img = [point[0] / w, point[1] / h]  # TODO: what is v_img?
    print("\nObject's v_img:\n", obj_v_img) if config.DEBUG else None

    return obj_v, obj_v_img


def generateSceneXML(h, w, quad_vn, path_env, path_albedo, path_rough, path_plane, path_obj, path_out="",
                     envmap_scale=1.0, obj_diffuse="#ffffff", obj_scale=1.0, obj_translate=(0.0, 0.0, 0.0),
                     sample=64, is_hdr=False):
    """
    generate XMLs of mitsuba renderer

    Args:
        h: height of image
        w: width of image
        quad_vn: normal of the plane (also the object)
        path_env: path to envmap
        path_albedo: path to albedo.png
        path_rough: path to roughness
        path_plane: path to plane.obj
        path_obj: path to shape of object
        path_out: path to output
        envmap_scale: scale of envmap irradiance,
        obj_diffuse: diffuse color of object inserted
        obj_scale: scale of object inserted
        obj_translate: translate of object inserted
        sample: sample count of renderer
        is_hdr: if true, render .exr; else .png

    Returns: None
    """
    # TODO: rotate envmap

    # set camera parameters
    camera_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    camera_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_trans = Transform(type="lookat", origin=camera_origin, target=camera_target, up=camera_up)

    # get shape of plane
    texture_albedo = Texture(name="diffuseReflectance", path_texture=path_albedo, scale=1.0)
    texture_rough = Texture(name="alpha", path_texture=path_rough, scale=1.0)
    plane_material = Material(bsdf="roughplastic", textures=[texture_albedo, texture_rough])
    plane_trans = None

    # get shape of object
    ## material
    obj_material = Material(bsdf="diffuse", srgb=obj_diffuse)
    ## transforms
    obj_trans = []
    ### scale parameters
    obj_scale = Transform(type="scale", x=obj_scale, y=obj_scale, z=obj_scale)
    obj_trans.append(obj_scale)
    ### rotate parameters
    rotate_axis = np.cross(camera_up, quad_vn.T.squeeze())
    if np.sum(rotate_axis * rotate_axis) > 1e-6:
        rotate_axis = rotate_axis / np.sqrt(np.sum(rotate_axis * rotate_axis))
        rotate_angle = np.arccos(np.sum(quad_vn.T.squeeze() * camera_up)) / np.pi * 180
        obj_rotate = Transform(type="rotate", x=rotate_axis[0], y=rotate_axis[1], z=rotate_axis[2], angle=rotate_angle)
        obj_trans.append(obj_rotate)
    ### translate parameters
    obj_translate = Transform(type="translate", x=obj_translate[0], y=obj_translate[1], z=obj_translate[2])
    obj_trans.append(obj_translate)

    # define material of mask
    mask_material = Material(bsdf="diffuse", srgb="#ffffff")

    # generate scene_all.xml (including object and plane)
    xml_handler = XmlHandler()
    xml_handler.addIntegrator()
    xml_handler.addSensor(fovX=config.FOV_X, transforms=[camera_trans], sample=sample, is_hdr=is_hdr, film=[w, h])
    xml_handler.addShape(type="obj", path_shape=path_plane, material=plane_material, transforms=plane_trans)
    # xml_handler.addShape(type="obj", path_shape=path_plane, material=obj_material, transforms=plane_trans)
    xml_handler.addShape(type="obj", path_shape=path_obj, material=obj_material, transforms=obj_trans)
    xml_handler.addEmitter(path_env=path_env, scale=envmap_scale)
    xml_handler.output2xml(path=osp.join(path_out, "scene_all.xml"))
    # generate scene_plane.xml (including plane only)
    xml_handler = XmlHandler()
    xml_handler.addIntegrator()
    xml_handler.addSensor(fovX=config.FOV_X, transforms=[camera_trans], sample=sample, is_hdr=is_hdr, film=[w, h])
    xml_handler.addShape(type="obj", path_shape=path_plane, material=plane_material, transforms=plane_trans)
    # xml_handler.addShape(type="obj", path_shape=path_plane, material=obj_material, transforms=plane_trans)
    xml_handler.addEmitter(path_env=path_env, scale=envmap_scale)
    xml_handler.output2xml(path=osp.join(path_out, "scene_plane.xml"))
    # generate mask_all.xml (including object and plane, output as mask)
    xml_handler = XmlHandler()
    xml_handler.addIntegrator()
    xml_handler.addSensor(fovX=config.FOV_X, transforms=[camera_trans], sample=sample, is_hdr=is_hdr, film=[w, h])
    xml_handler.addShape(type="obj", path_shape=path_plane, material=mask_material, transforms=plane_trans)
    xml_handler.addShape(type="obj", path_shape=path_obj, material=mask_material, transforms=obj_trans)
    xml_handler.addEmitter(directional=[0, 0, 1], scale=100.0)
    xml_handler.addEmitter(directional=[0, 0, -1], scale=100.0)
    xml_handler.addEmitter(directional=[1, 0, 0], scale=100.0)
    xml_handler.addEmitter(directional=[-1, 0, 0], scale=100.0)
    xml_handler.output2xml(path=osp.join(path_out, "mask_all.xml"))
    # generate mask_object.xml (including obj only, output as mask)
    xml_handler = XmlHandler()
    xml_handler.addIntegrator()
    xml_handler.addSensor(fovX=config.FOV_X, transforms=[camera_trans], sample=sample, is_hdr=is_hdr, film=[w, h])
    xml_handler.addShape(type="obj", path_shape=path_obj, material=mask_material, transforms=obj_trans)
    xml_handler.addEmitter(directional=[0, 0, 1], scale=100.0)
    xml_handler.addEmitter(directional=[0, 0, -1], scale=100.0)
    xml_handler.addEmitter(directional=[1, 0, 0], scale=100.0)
    xml_handler.addEmitter(directional=[-1, 0, 0], scale=100.0)
    xml_handler.output2xml(path=osp.join(path_out, "mask_object.xml"))


def differentialRender(path_mitsuba, path_img_in, path_img_out, path_out, is_process=True):
    os.system("{} -o {} {}".format(path_mitsuba, osp.join(path_out, "scene_all.png"), osp.join(path_out, "scene_all.xml")))
    os.system("{} -o {} {}".format(path_mitsuba, osp.join(path_out, "scene_plane.png"), osp.join(path_out, "scene_plane.xml")))
    os.system("{} -o {} {}".format(path_mitsuba, osp.join(path_out, "mask_all.png"), osp.join(path_out, "mask_all.xml")))
    os.system("{} -o {} {}".format(path_mitsuba, osp.join(path_out, "mask_object.png"), osp.join(path_out, "mask_object.xml")))

    image_origin = cv2.imread(osp.join(path_img_in))
    scene_all = cv2.imread(osp.join(path_out, "scene_all.png"))
    scene_plane = cv2.imread(osp.join(path_out, "scene_plane.png"))
    mask_all = cv2.imread(osp.join(path_out, "mask_all.png"))
    mask_object = cv2.imread(osp.join(path_out, "mask_object.png"))
    mask_background = mask_all - mask_object

    def processMask(_mask, _gb=(1, 1), _mb=3, _flag=is_process, _is_erode=True):
        _mask = cv2.cvtColor(_mask.astype('uint8'), cv2.COLOR_BGR2GRAY)
        if _is_erode:
            _mask = cv2.erode(_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) if _flag else _mask
        else:
            _mask = cv2.dilate(_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) if _flag else _mask
        _mask = cv2.GaussianBlur(_mask, _gb, 0) if _flag else _mask
        _mask = cv2.medianBlur(_mask, _mb) if _flag else _mask
        _mask = np.repeat(_mask[..., np.newaxis], 3, 2)
        return _mask

    mask_all = processMask(mask_all) / 255.0

    mask_object = processMask(mask_object) / 255.0

    mask_background = np.maximum(0.0, mask_background)
    mask_background = processMask(mask_background) / 255.0
    mask_ratio = scene_all / np.maximum(1e-10, scene_plane)
    # mask_ratio = scene_all / np.maximum(1e-10, scene_plane) * mask_background
    mask_ratio = np.maximum(0.0, mask_ratio)
    mask_ratio = np.minimum(1.0, mask_ratio)

    image_inserted = (1 - mask_background) * image_origin + mask_background * image_origin * mask_ratio
    # image_inserted = (1 - mask_all) * image_origin + mask_all * image_origin * mask_ratio
    image_inserted = (1 - mask_object) * image_inserted + mask_object * scene_all

    def getEdge(_mask, _ek=(5, 5), _dk=(5, 5)):
        _mask_erode = cv2.erode(_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _ek))
        _mask_dilate = cv2.dilate(_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _dk))
        _edge = _mask_dilate - _mask_erode
        return _edge
    if is_process:
        edge_object = getEdge(mask_object, _ek=(3, 3), _dk=(9, 9))
        edge_background = getEdge(mask_background, _ek=(5, 5), _dk=(9, 9))
        edge_and = cv2.bitwise_and(edge_object, edge_background)
        # cv2.imshow("edge_object", edge_object * 255)
        # cv2.imshow("edge_background", edge_background * 255)
        _, edge_and = cv2.threshold(edge_and, 1e-5, 1, cv2.THRESH_BINARY)
        edge_and = processMask(edge_and, _gb=(13, 13), _mb=3)
        # cv2.imshow("edge_and", edge_and * 255)
        # cv2.imshow("1-edge_and", (1 - edge_and) * 255)
        image_inserted = (1 - edge_and) * image_inserted + edge_and * image_origin * mask_ratio
    # cv2.imshow("image_inserted", image_inserted)
    # cv2.waitKey(0)

    # image_inserted_object = np.where(mask_object > 0, scene_all, 255.0)
    # image_inserted_object = cv2.dilate(image_inserted_object, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # image_inserted_object = np.where(image_inserted_object == np.array([255, 255, 255]), 0, image_inserted_object)

    # image_inserted_object = utils.blurImageEdge(image_inserted_object, ksize=(5, 5), offset=2)

    # mask_all = processMask(mask_all, _ksize=5) / 255.0
    # mask_object = processMask(mask_object, _ksize=0) / 255.0
    # mask_background = np.maximum(0.0, mask_all - mask_object)
    #
    # mask_ratio = scene_all / np.maximum(1e-10, scene_plane)
    # mask_ratio = np.maximum(0.0, mask_ratio)
    # mask_ratio = np.minimum(1.0, mask_ratio)
    #
    # image_background = image_origin * mask_ratio
    # # image_background = utils.blurImageEdge(image_background)
    # image_object = scene_all * mask_object
    # image_object = utils.blurImageEdge(image_object, ksize=(5, 5), offset=2)
    # mask_object = utils.blurImageEdge(mask_object, ksize=(5, 5), offset=5)
    # image_object = image_object * mask_object
    #
    # grey_background = processMask(image_background, _ksize=0)
    # grey_object = processMask(image_object, _ksize=0)
    #
    # image_inserted_object = np.where(grey_object > 10, image_object, image_origin)
    # image_inserted_background = np.where(grey_background > 10, image_background, image_origin)
    #
    # # image_inserted = image_origin * (1 - mask_background) + image_background * mask_background
    # # image_inserted = image_inserted * (1 - mask_object) + image_object * mask_object
    # image_inserted = np.where(grey_background > 10, image_background, image_origin)
    # image_inserted = np.where(grey_object > 45, image_object, image_inserted)
    #
    # cv2.imwrite(osp.join(path_out, "image_object.png"), image_object) if config.DEBUG else None
    # cv2.imwrite(osp.join(path_out, "image_background.png"), image_background) if config.DEBUG else None
    # cv2.imwrite(osp.join(path_out, "image_inserted_object.png"), image_inserted_object) if config.DEBUG else None
    # cv2.imwrite(osp.join(path_out, "image_inserted_background.png"), image_inserted_background) if config.DEBUG else None
    cv2.imwrite(path_img_out, image_inserted)
    print("Image output to =>", path_img_out)


if __name__ == '__main__':
    information = pd.read_csv("data/dataset/render.csv", index_col=0)
    for index, info in information.iterrows():
        dir_name = osp.splitext(osp.basename(info['GT']))[0]

        if info['Ours'] not in ["Ours\\scene01_probe02.exr"]:
            continue

        dir_out = osp.join("data", "outputs", dir_name)
        os.system("mkdir {}".format(dir_out)) if not osp.isdir(dir_out) else None

        image = Image.open(osp.join("data/dataset", info['path_im']))
        height, width = image.size[1], image.size[0]
        obj2D = np.asarray(eval(info['position']))
        quad2D = point2quadrangle(height, width, point=obj2D, offset=5)
        mask2D = generatePlaneMask(height, width, quad2D)
        vn3D, v3D, _ = generatePlane3D(height, width, quad2D, mask2D, path_normal=osp.join("data/dataset", info['path_normal']), path_plane="")
        obj_v3D, _ = generateObject3D(height, width, point=obj2D, quad=quad2D, quad_vn=vn3D, quad_v=v3D)
        plane2D = point2quadrangle(height, width, point=obj2D, offset=60)
        plane_mask2D = generatePlaneMask(height, width, plane2D)
        _, plane3D, _ = generatePlane3D(height, width, plane2D, plane_mask2D, plane_vn=vn3D,
                                        path_normal=osp.join("data/dataset", info['path_normal']), path_plane=osp.join(dir_out, "plane.obj"))

        for paper in ["GT", "Gardner17", "Gardner19", "Garon19", "Li20", "Ours"]:
            dir_paper_out = osp.join(dir_out, paper)
            os.system("mkdir {}".format(dir_paper_out)) if not osp.isdir(dir_paper_out) else None

            generateSceneXML(
                height, width, vn3D,
                path_env=osp.join("data/dataset", info[paper]),
                path_albedo=osp.join("data/dataset", info['path_albedo']),
                path_rough=osp.join("data/dataset", info['path_rough']),
                path_plane=osp.join(dir_out, "plane.obj"),
                path_obj=config.PATH_DATA_OBJECT,
                path_out=dir_paper_out,
                envmap_scale=1.0,
                obj_diffuse="#ffffff",
                obj_scale=0.8,
                obj_translate=0.94*obj_v3D,
                sample=1024,
                is_hdr=False,
            )
            differentialRender(path_mitsuba=config.PATH_MITSUBA, path_img_in=osp.join("data/dataset", info['path_im']),
                               path_img_out=osp.join(dir_paper_out, "{}_{}.png".format(dir_name, paper)), path_out=dir_paper_out)
