import os
import xml.dom.minidom
import h5py

if __name__ == '__main__':
    rpath = './data/train/gt'
    fnames = os.listdir(rpath)
    for n in fnames:
        name = n.replace('.xml', '')
        fpath = os.path.join(rpath, n)
        dom = xml.dom.minidom.parse(fpath)
        root = dom.documentElement
        obj = root.getElementsByTagName('object').length
        hpath = os.path.join(rpath, n)
        hf = h5py.File(hpath.replace('.xml', '.h5'), 'w')
        hf['gt_count'] = obj
        print(obj)

