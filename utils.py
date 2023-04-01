import math
import os.path as osp
import cv2
import numpy as np
import imageio

import config


def isPointInQuadrangle(point, quad) -> bool:
    """
    judge if point in the quadrangle

    Args:
        point:  [x,y] (np.array)
        quad:   [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (np.array)

    Returns:
        True if point in the quadrangle, else False
    """
    flag = True
    for i in range(4):
        vec_q = np.asarray(quad[(i + 1) % 4] - quad[i])
        vec_qp = np.asarray(point - quad[i])
        re = np.cross(np.append(vec_q, 0), np.append(vec_qp, 0))
        flag = flag and (re[2] < 0)
    return flag


def point2quadrangle(h, w, point, offset):
    """
        generate the quadrangle around the input point

        Args:
            h: height of image
            w: width of image
            point:  [x,y] (np.array)
            offset: offset of the quadrangle

        Returns:
            quad:   [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (np.array)
        """
    quad = np.asarray([
        [point[0] - 3 * offset, point[1] - 1 * offset],
        [point[0] - 1 * offset, point[1] + 2 * offset],
        [point[0] + 3 * offset, point[1] + 1 * offset],
        [point[0] + 1 * offset, point[1] - 2 * offset],
    ])
    quad = np.maximum(1, quad)
    for point in quad:
        point[0] = np.minimum(w, point[0])
        point[1] = np.minimum(w, point[1])
    return quad


def output2obj(path, vn, v, vt, mode="quad"):
    """
    output the information of objects to .obj file

    Args:
        path: output path
        vn: normal of plane - np.array(3,1)
        v: 4 vertices of plane - np.array(4,3)
        vt: texture coordinates of each vertex - np.array(4,2)
        mode: output mode

    Returns:
        None
    """
    if mode == "quad":
        with open(path, 'w') as f:
            for i in range(4):
                f.write('v %.5f %.5f %.5f\n' % (v[i][0], v[i][1], v[i][2]))
            for i in range(4):
                f.write('vt %.5f %.5f\n' % (vt[i][0], vt[i][1]))
            f.write('vn %.5f %.5f %.5f\n' % (vn[0], vn[1], vn[2]))
            f.write('f 1/1/1 2/2/1 3/3/1\n')
            f.write('f 1/1/1 3/3/1 4/4/1\n')
    else:
        print("function output2obj() wrong!")
        exit(-1)


def exr2ldr(path_exr):
    """
    transfer .exr to ldr mat

    Args:
        path_exr: path to .exr image

    Returns:
        ldr transferred array
    """
    exr = imageio.imread(path_exr)
    exr_gamma_correct = np.clip(np.power(exr, 0.45), 0, 1)
    ldr = np.asarray(np.uint8(exr_gamma_correct * 255))
    return ldr


def hdr2ldr(path_hdr):
    """
    transfer .hdr to ldr mat
    (refer to https://matiascodesal.com/blog/how-convert-hdr-image-ldr-using-python/)

    Args:
        path_hdr: path to .hdr image

    Returns:
        ldr transferred array
    """
    hdr = cv2.imread(path_hdr, flags=cv2.IMREAD_ANYDEPTH)
    # Tone-mapping and color space conversion
    tonemap = cv2.createTonemapDrago(2.2)
    scale = 5
    ldr = scale * tonemap.process(hdr)
    # Remap to 0-255 for the bit-depth conversion
    ldr = ldr * 255
    return ldr


def spherical2latitude(h, w, envmap_spherical, path_out="envmap_latitude.png", rotate_h=0, rotate_w=0):
    def sampleSphere3D(_r=1, _size=100):
        _chi_x = np.random.random(_size)
        _chi_y = np.random.random(_size)
        _theta = np.arccos(1 - 2 * _chi_x)
        _phi = 2 * np.pi * _chi_y
        _x = _r * np.sin(_theta) * np.cos(_phi)
        _y = _r * np.sin(_theta) * np.sin(_phi)
        _z = _r * np.cos(_theta)
        return _x, _y, _z

    def getRFromLatitudeUV(_u, _v):
        _Rx_div_Rz = np.tan(_u * 2 * np.pi - np.pi)
        _Ry = np.sin(_v * np.pi - np.pi / 2)
        _Rz = (1 - _Ry ** 2) / (1 + _Rx_div_Rz ** 2)
        _Rz = _Rz if 0.25 < _u < 0.75 else - _Rz
        _Rx = _Rz * _Rx_div_Rz
        return _Rx, _Ry, _Rz

    def getSphericalUVFromR(_Rx, _Ry, _Rz):
        _m = 2 * np.sqrt(_Rx ** 2 + _Ry ** 2 + (_Rz + 1) ** 2)
        _u = _Rx / _m + 0.5
        _v = _Ry / _m + 0.5
        return _u, _v

    def getLatitudeUVFromR(_Rx, _Ry, _Rz,  _rotate_h=rotate_h, _rotate_w=90+rotate_w):
        _rotate_h = _rotate_h / 180.0 * np.pi
        _m_rotate_h = np.asarray([[np.cos(_rotate_h), np.sin(_rotate_h)], [-np.sin(_rotate_h), np.cos(_rotate_h)]])
        _Ry_Rz = np.asarray([_Ry, _Rz])
        _Ry_Rz = _m_rotate_h @ _Ry_Rz
        _Ry = _Ry_Rz[0, :]
        _Rz = _Ry_Rz[1, :]

        _rotate_w = _rotate_w / 180.0 * np.pi
        _m_rotate_w = np.asarray([[np.cos(_rotate_w), np.sin(_rotate_w)], [-np.sin(_rotate_w), np.cos(_rotate_w)]])
        _Rx_Rz = np.asarray([_Rx, _Rz])
        _Rx_Rz = _m_rotate_w @ _Rx_Rz
        _Rx = _Rx_Rz[0, :]
        _Rz = _Rx_Rz[1, :]

        arctan_Rx_div_Rz = np.arctan(_Rx / _Rz)
        arctan_Rx_div_Rz = np.where(_Rz > 0, arctan_Rx_div_Rz - np.pi / 2, arctan_Rx_div_Rz + np.pi / 2)

        _u = (arctan_Rx_div_Rz + np.pi) / (2 * np.pi)
        _v = (np.arcsin(_Ry) + np.pi / 2) / np.pi
        return _u, _v

    def getTextureFromSpherical(_u, _v, _envmap):
        _h, _w = _envmap.shape[0], _envmap.shape[1]
        _texture = _envmap[int((1 - _v) * _h), int(_u * _w), :]
        # envmap_spherical_copy[int((1 - _v) * _h), int(_u * _w), :] = 255
        return _texture

    # envmap_spherical_copy = np.copy(envmap_spherical)
    envmap_latitude = np.zeros((h, w, 4)) if path_out.endswith(".exr") else np.zeros((h, w, 3))
    Rx, Ry, Rz = sampleSphere3D(_r=1, _size=1024 * 1024)
    u_S, v_S = getSphericalUVFromR(Rx, Ry, Rz)
    u_L, v_L = getLatitudeUVFromR(Rx, Ry, Rz)
    for us, vs, ul, vl in zip(u_S, v_S, u_L, v_L):
        envmap_latitude[int((1 - vl) * h), int(ul * w), :] = getTextureFromSpherical(us, vs, envmap_spherical)

    if path_out.endswith(".png"):
        cv2.imwrite(path_out, envmap_latitude)
        # cv2.imwrite(osp.join(path_out, "envmap_spherical.{}".format(suffix)), envmap_spherical_copy)
    elif path_out.endswith(".exr"):
        imageio.imwrite(path_out, np.asarray(envmap_latitude, dtype="float32"), format="exr")

    print("Envmap output to =>", path_out)


def removeOutlier(mat):
    array = np.copy(mat).flatten()
    array = array[abs(array) > 1e-8]
    percentile = np.percentile(array, [0, 5, 50, 95, 100])
    IQR = percentile[3] - percentile[1]
    limit_up = percentile[3] + IQR * 1.5
    limit_down = percentile[1] - IQR * 1.5
    mat[mat > limit_up] = 1.0
    mat[mat < limit_down] = 0.0
    return mat


def blurImageEdge(image, ksize=(21, 21), offset=2):
    gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, np.uint8)
    _, thresh = cv2.threshold(gray, 5, 250, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), offset)

    image_blurred = cv2.GaussianBlur(image, ksize, 0)
    image_result = np.where(mask == np.array([255, 255, 255]), image_blurred, image)

    cv2.imwrite("output.jpg", image_result)
    return image_result


def scaleEnvmap(path_envmap, path_envmap_out):
    envmap = imageio.imread(path_envmap, format="exr")
    envmap = (np.sin(envmap * np.pi - np.pi / 2) + 1) / 2
    imageio.imwrite(path_envmap_out, np.asarray(envmap, dtype="float32"), format="exr")


if __name__ == '__main__':
    # envmap = cv2.imread("gl_map.jpg")
    # envmap = imageio.imread("scene00_probe00.exr", format="exr")
    # spherical2latitude(256, 512, envmap,
    #                    path_out="scene00_probe00_envmap.exr",
    #                    rotate_h=int((3840/2-2797)/3840*config.FOV_X*(3840/5760)),
    #                    rotate_w=-int((5760/2-614)/5760*config.FOV_X))
    scaleEnvmap("data/debug/scene01/scene01_probe02.jpgbottom_164_193.exr", "data/debug/scene01/scene01_probe02.reshape02.exr")
    scaleEnvmap("data/debug/scene01/scene01_probe02.reshape02.exr", "data/debug/scene01/scene01_probe02.reshape03.exr")