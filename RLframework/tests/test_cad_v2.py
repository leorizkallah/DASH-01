"""Physical-property regression checks; run with unittest from repository root."""
import json
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from RLframework.model import prepare_cad_v2 as cad
from controller.identification.apply_identified import apply_body_inertials


class InertialMathTests(unittest.TestCase):
    def setUp(self):
        self.frame = {"rotation_x_deg": 0, "translation_mm": [0, 0, 0]}
        self.part = {
            "mass_g": 2000, "com_mm": [100, 0, 0],
            "inertia_com_g_mm2": dict(zip(cad.ORDER, [2e6, 3e6, 4e6, 0, 0, 0])),
            "inertia_origin_g_mm2": dict(zip(cad.ORDER, [2e6, 23e6, 24e6, 0, 0, 0])),
        }

    def test_units_and_payload_distribution(self):
        result, _ = cad.convert(self.part, self.frame, extra_kg=1)
        self.assertEqual(result["mass"], 3)
        np.testing.assert_allclose(result["com"], [.1, 0, 0])
        np.testing.assert_allclose([result["inertia"][k] for k in ("ixx", "iyy", "izz")], [.003, .0045, .006])

    def test_com_frame_transform_does_not_double_apply_parallel_axis(self):
        result, _ = cad.convert(self.part, {"rotation_x_deg": 90, "translation_mm": [0, 10, 20]})
        np.testing.assert_allclose(result["com"], [.1, .01, .02])
        np.testing.assert_allclose([result["inertia"][k] for k in ("ixx", "iyy", "izz")], [.002, .004, .003])

    def test_bad_tensor_and_bad_payload_are_rejected(self):
        for tensor in (np.diag([-1, 1, 1]), np.diag([1, 1, 3])):
            with self.assertRaises(ValueError):
                cad.validate_inertia(1, np.zeros(3), tensor)
        for extra in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cad.convert(self.part, self.frame, extra)

    def test_wrong_pdf_origin_tensor_is_rejected(self):
        self.part["inertia_origin_g_mm2"]["Iyy"] = 3e6
        with self.assertRaises(ValueError):
            cad.convert(self.part, self.frame)

    def test_strict_mapping_and_stale_inertial_orientation(self):
        root = ET.fromstring('<mujoco><worldbody><body name="link"><inertial mass="1" pos="0 0 0" diaginertia="1 1 1" euler="1 0 0"/></body></worldbody></mujoco>')
        bp, _ = cad.convert(self.part, self.frame)
        with self.assertRaises(ValueError):
            apply_body_inertials(root, {"typo": bp}, strict=True)
        apply_body_inertials(root, {"link": bp}, strict=True)
        inertial = root.find(".//inertial")
        self.assertNotIn("euler", inertial.attrib)
        self.assertNotIn("diaginertia", inertial.attrib)
        self.assertIn("fullinertia", inertial.attrib)


class CadIntegrationTests(unittest.TestCase):
    def test_build_mass_preservation_and_repeatability(self):
        original = cad.digest(cad.SOURCE)
        xml, audit = cad.build()
        no_payload_xml, no_payload = cad.build(extra_body_kg=0)
        repeated, _ = cad.build()
        self.assertEqual(xml, repeated)
        self.assertEqual(original, cad.digest(cad.SOURCE))
        self.assertAlmostEqual(audit["corrected_mass_kg"], 14.594154, places=8)
        self.assertAlmostEqual(no_payload["corrected_mass_kg"], 13.594154, places=8)
        self.assertEqual(len(audit["bodies"]), 15)
        self.assertLess(max(audit["initial_loop_gap_mm"].values()), .001)
        self.assertFalse(audit["training_ready"])
        self.assertEqual(audit["model_dimensions"]["neq"], 4)
        # Only inertials/compiler/comment change; joint limits and topology survive.
        raw = ET.parse(cad.SOURCE).getroot()
        corrected = ET.fromstring(xml)
        for tag in ("joint", "body", "site", "geom", "connect"):
            self.assertEqual([e.attrib for e in raw.iter(tag)], [e.attrib for e in corrected.iter(tag)])

    def test_confirmed_frames_are_identity_and_pdf_values_used_directly(self):
        parts = json.loads(cad.PROPERTIES.read_text())["parts"]
        mapping = json.loads(cad.MAPPING.read_text())["bodies"]
        for frame in mapping.values():
            self.assertEqual(frame["rotation_x_deg"], 0)
            self.assertEqual(frame["translation_mm"], [0, 0, 0])
            part = parts[frame["part"]]
            bp, _ = cad.convert(part, frame)
            np.testing.assert_allclose(bp["com"], np.asarray(part["com_mm"]) * 1e-3)
            np.testing.assert_allclose(list(bp["inertia"].values()), [part["inertia_com_g_mm2"][k]*1e-9 for k in cad.ORDER])


if __name__ == "__main__":
    unittest.main()
