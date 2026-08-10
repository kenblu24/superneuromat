import unittest
import pathlib as pl
import sys
sys.path.insert(0, "../../src/")


cwd = pl.Path(__file__).parent


class CaspianTest(
    unittest.TestCase,
):
    def test_import_caspian_json(self):
        """ Test import caspian
        """
        from superneuromat.port.caspian import CaspianImporter

        assert pl.Path(cwd / 'example.json').exists()

        importer = CaspianImporter(cwd / 'example.json')
        import json
        with open(cwd / 'example.json') as f:
            j = json.load(f)
        snn = importer.network_from_json(j)
        print(snn)

