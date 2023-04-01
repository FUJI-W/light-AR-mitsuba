import xml.etree.ElementTree as et
from xml.dom import minidom
import config


class Texture:
    def __init__(self, name, path_texture, scale=1.0):
        """
        texture in mitsuba
        Args:
            name: type/parameter of texture
            path_texture: path to the texture file
            scale: scale of intensity
        """
        self.name = name
        self.path_texture = path_texture
        self.scale = scale


class Material:
    def __init__(self, bsdf="diffuse", srgb=None, textures=None):
        """
        material in mitsuba
        Args:
            bsdf: type of material, diffuse/roughplastic
            srgb: when bsdf is diffuse, set the color
            textures: when bsdf is roughplastic, set the parameters (a list of Texture())
        """
        self.type = bsdf
        self.srgb = srgb
        self.textures = textures


class Transform:
    def __init__(self, type="scale", x=0.0, y=0.0, z=0.0, angle=None, origin=None, target=None, up=None):
        """
        transforms in mitsuba
        Args:
            type: scale; translate; rotate; lookat
            x: value of x axis
            y: value of y axis
            z: value of y axis
            angle: when rotate, angle is clockwise angle
            origin: when lookAt, the origin position
            target: when lookAt, the lookat point
            up: when lookAt, the up vector of camera
        """
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle
        self.origin = origin
        self.target = target
        self.up = up


class XmlHandler:
    def __init__(self):
        self.root = et.Element('scene')
        self.root.set('version', '0.5.0')

    def addIntegrator(self, integrator="path", max_depth=-1):
        sub = et.SubElement(self.root, 'integrator')
        sub.set('type', integrator)
        sub_sub = et.SubElement(sub, 'integer')
        sub_sub.set('name', 'maxDepth')
        sub_sub.set('value', str(max_depth))

    def addShape(self, type, path_shape, material, transforms):
        sub = et.SubElement(self.root, 'shape')
        sub.set('type', type)
        # add path of shape
        sub_sub = et.SubElement(sub, 'string')
        sub_sub.set('name', 'filename')
        sub_sub.set('value', path_shape)
        # add material(bsdf)
        sub_sub = et.SubElement(sub, 'bsdf')
        sub_sub.set('type', material.type)
        if material.srgb is not None:
            sub_sub_sub = et.SubElement(sub_sub, 'srgb')
            sub_sub_sub.set('name', 'reflectance')
            sub_sub_sub.set('value', material.srgb)
        if material.textures is not None:
            for tex in material.textures:
                sub_sub_sub = et.SubElement(sub_sub, 'texture')
                sub_sub_sub.set('name', tex.name)
                sub_sub_sub.set('type', 'scale')
                sub_sub_sub_sub = et.SubElement(sub_sub_sub, 'texture')
                sub_sub_sub_sub.set('name', tex.name)
                sub_sub_sub_sub.set('type', 'bitmap')
                sub_sub_sub_sub_sub = et.SubElement(sub_sub_sub_sub, 'string')
                sub_sub_sub_sub_sub.set('name', 'filename')
                sub_sub_sub_sub_sub.set('value', tex.path_texture)
                sub_sub_sub_sub_sub = et.SubElement(sub_sub_sub_sub, 'float')
                sub_sub_sub_sub_sub.set('name', 'scale')
                sub_sub_sub_sub_sub.set('value', str(tex.scale))
        # add transforms
        sub_sub = et.SubElement(sub, 'transform')
        sub_sub.set('name', 'toWorld')
        if transforms is not None:
            for trans in transforms:
                if trans.type == "scale":
                    sub_sub_sub = et.SubElement(sub_sub, 'scale')
                    sub_sub_sub.set('x', str(trans.x))
                    sub_sub_sub.set('y', str(trans.y))
                    sub_sub_sub.set('z', str(trans.z))
                elif trans.type == "translate":
                    sub_sub_sub = et.SubElement(sub_sub, 'translate')
                    sub_sub_sub.set('x', str(trans.x))
                    sub_sub_sub.set('y', str(trans.y))
                    sub_sub_sub.set('z', str(trans.z))
                elif trans.type == "rotate":
                    sub_sub_sub = et.SubElement(sub_sub, 'rotate')
                    sub_sub_sub.set('x', str(trans.x))
                    sub_sub_sub.set('y', str(trans.y))
                    sub_sub_sub.set('z', str(trans.z))
                    sub_sub_sub.set('angle', str(trans.angle))
                elif trans.type == "lookat":
                    sub_sub_sub = et.SubElement(sub_sub, 'lookat')
                    sub_sub_sub.set('origin', "%.5f %.5f %.5f" % (trans.origin[0], trans.origin[1], trans.origin[2]))
                    sub_sub_sub.set('target', "%.5f %.5f %.5f" % (trans.target[0], trans.target[1], trans.target[2]))
                    sub_sub_sub.set('up', "%.5f %.5f %.5f" % (trans.up[0], trans.up[1], trans.up[2]))

    def addEmitter(self, directional=None, path_env="", scale=1.0):
        if directional is not None:
            sub = et.SubElement(self.root, 'emitter')
            sub.set('type', 'directional')
            sub_sub = et.SubElement(sub, "vector")
            sub_sub.set("name", "direction")
            sub_sub.set("x", str(directional[0]))
            sub_sub.set("y", str(directional[1]))
            sub_sub.set("z", str(directional[2]))
            sub_sub = et.SubElement(sub, "spectrum")
            sub_sub.set("name", "irradiance")
            sub_sub.set("value", str(scale))
        else:
            sub = et.SubElement(self.root, 'emitter')
            sub.set('type', 'envmap')
            # add path to envmap
            sub_sub = et.SubElement(sub, 'string')
            sub_sub.set('name', 'filename')
            sub_sub.set('value', path_env)
            # add envmap scale
            sub_sub = et.SubElement(sub, 'float')
            sub_sub.set('name', 'scale')
            sub_sub.set('value', str(scale))

    def addSensor(self, fovX, transforms, sample=64, is_hdr=False, film=(480, 480)):
        sub = et.SubElement(self.root, 'sensor')
        sub.set('type', 'perspective')
        # add fov of x axis
        sub_sub = et.SubElement(sub, 'string')
        sub_sub.set('name', 'fovAxis')
        sub_sub.set('value', 'x')
        sub_sub = et.SubElement(sub, 'float')
        sub_sub.set('name', 'fov')
        sub_sub.set('value', str(fovX))
        # add transforms
        sub_sub = et.SubElement(sub, 'transform')
        sub_sub.set('name', 'toWorld')
        for trans in transforms:
            if trans.type == "scale":
                sub_sub_sub = et.SubElement(sub_sub, 'scale')
                sub_sub_sub.set('x', str(trans.x))
                sub_sub_sub.set('y', str(trans.y))
                sub_sub_sub.set('z', str(trans.z))
            elif trans.type == "translate":
                sub_sub_sub = et.SubElement(sub_sub, 'translate')
                sub_sub_sub.set('x', str(trans.x))
                sub_sub_sub.set('y', str(trans.y))
                sub_sub_sub.set('z', str(trans.z))
            elif trans.type == "rotate":
                sub_sub_sub = et.SubElement(sub_sub, 'rotate')
                sub_sub_sub.set('x', str(trans.x))
                sub_sub_sub.set('y', str(trans.y))
                sub_sub_sub.set('z', str(trans.z))
                sub_sub_sub.set('angle', str(trans.angle))
            elif trans.type == "lookat":
                sub_sub_sub = et.SubElement(sub_sub, 'lookat')
                sub_sub_sub.set('origin', "%.5f %.5f %.5f" % (trans.origin[0], trans.origin[1], trans.origin[2]))
                sub_sub_sub.set('target', "%.5f %.5f %.5f" % (trans.target[0], trans.target[1], trans.target[2]))
                sub_sub_sub.set('up', "%.5f %.5f %.5f" % (trans.up[0], trans.up[1], trans.up[2]))
        # add sample count
        sub_sub = et.SubElement(sub, 'sampler')
        sub_sub.set('type', 'independent')
        sub_sub_sub = et.SubElement(sub_sub, 'integer')
        sub_sub_sub.set('name', 'sampleCount')
        sub_sub_sub.set('value', str(sample))
        # add film
        sub_sub = et.SubElement(sub, 'film')
        sub_sub.set('type', 'hdrfilm' if is_hdr else 'ldrfilm')
        sub_sub_sub = et.SubElement(sub_sub, 'integer')
        sub_sub_sub.set('name', 'width')
        sub_sub_sub.set('value', str(film[0]))
        sub_sub_sub = et.SubElement(sub_sub, 'integer')
        sub_sub_sub.set('name', 'height')
        sub_sub_sub.set('value', str(film[1]))

    def output2xml(self, path):
        string_origin = et.tostring(self.root, 'utf-8')
        string_parse = minidom.parseString(string_origin)
        string_xml = string_parse.toprettyxml(indent="    ")
        string_xml = string_xml.split('\n')
        string_xml = [x for x in string_xml if len(x.strip()) != 0]
        string_xml = '\n'.join(string_xml)
        with open(path, 'w') as f:
            f.write(string_xml)
        print("XML output to =>", path) if config.DEBUG else None
