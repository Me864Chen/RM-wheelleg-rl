"""The standalone variants must contain the current raw export's physical data."""
import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def without_mesh_paths(path):
    tree=ET.parse(path).getroot()
    for mesh in tree.findall('.//mesh'):
        mesh.set('filename','')
    return ET.tostring(tree)


class ModelReplacementTests(unittest.TestCase):
    def test_current_export_replaces_the_old_raw_urdf(self):
        files=list((ROOT/'heroleg26/urdf').glob('*.urdf'))
        self.assertEqual([p.name for p in files], ['4.SLDASM.urdf'])
        self.assertEqual(ET.parse(files[0]).getroot().get('name'), '4.SLDASM')
        self.assertFalse((ROOT/'heroleg26/urdf/2.SLDASM.urdf').exists())
        self.assertFalse(list((ROOT/'heroleg26').rglob('*.bak*')))

    def test_all_variants_preserve_current_physical_xml(self):
        reference=without_mesh_paths(ROOT/'heroleg26/urdf/4.SLDASM.urdf')
        for path in ['heroleg26_sim/urdf/heroleg26.urdf',
                     'heroleg26_sim/urdf/heroleg26_visual.urdf']:
            self.assertEqual(without_mesh_paths(ROOT/path), reference)
            for mesh in ET.parse(ROOT/path).getroot().findall('.//mesh'):
                self.assertNotIn('://', mesh.get('filename'))
                self.assertTrue(((ROOT/path).parent/mesh.get('filename')).resolve().is_file())

    def test_manifest_and_new_mesh_copies_match(self):
        sim=ROOT/'heroleg26_sim'
        data=json.loads((sim/'resource_manifest.json').read_text())
        self.assertEqual(data['source_urdf'], '../heroleg26/urdf/4.SLDASM.urdf')
        self.assertEqual(len(data['mesh_resources']),7)
        for record in data['mesh_resources']:
            actual=sim/record['path']
            raw=ROOT/'heroleg26/meshes'/actual.name
            for path in (raw,actual):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])


if __name__=='__main__':
    unittest.main()
