"""Dependency-free regression checks for implicit MJLab robot bindings."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "src" / "wheelleg"


class EntityBindingTests(unittest.TestCase):
    def test_builtin_terms_have_explicit_asset(self):
        tree = ast.parse((ROOT / "config/base_env_cfg.py").read_text(encoding="utf-8"))
        targets = {"base_ang_vel", "projected_gravity", "joint_torques_l2",
                   "joint_acc_l2", "joint_pos_limits", "bad_orientation"}
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            func = keywords.get("func")
            if not isinstance(func, ast.Attribute) or func.attr not in targets:
                continue
            seen.add(func.attr)
            params = keywords.get("params")
            self.assertIsInstance(params, ast.Dict, func.attr)
            bindings = {key.value: value for key, value in zip(params.keys, params.values)
                        if isinstance(key, ast.Constant)}
            asset = bindings.get("asset_cfg")
            self.assertIsInstance(asset, ast.Call, func.attr)
            self.assertEqual(asset.args[0].value, "wheelleg", func.attr)
        self.assertEqual(seen, targets)

    def test_critic_wrapper_binds_asset(self):
        tree = ast.parse((ROOT / "mdp/rewards.py").read_text(encoding="utf-8"))
        wrapper = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "safe_base_lin_vel")
        call = next(node for node in ast.walk(wrapper)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "base_lin_vel")
        asset = next(kw.value for kw in call.keywords if kw.arg == "asset_cfg")
        self.assertEqual(asset.args[0].value, "wheelleg")


if __name__ == "__main__":
    unittest.main()
