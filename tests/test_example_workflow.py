import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "HR-Endless-Sampler-Example.json"


class ExampleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def input_link(self, node_id, name):
        node = self.nodes[node_id]
        return next(item["link"] for item in node["inputs"] if item["name"] == name)

    def test_prompt_skill_planned_frames_drive_conditioning_length(self):
        link_id = self.input_link(2603, "length")
        link = self.links[link_id]
        self.assertEqual(link[1:5], [2602, 5, 2603, 6])
        self.assertEqual(link[5], "INT")

    def test_prompt_plan_and_event_ledger_drive_sampler(self):
        prompt = self.links[self.input_link(2576, "prompt")]
        ledger = self.links[self.input_link(2576, "initial_event_ledger")]
        plan = self.links[self.input_link(2576, "prompt_plan")]
        self.assertEqual(prompt[1:4], [2602, 0, 2576])
        self.assertEqual(ledger[1:4], [2602, 2, 2576])
        self.assertEqual(plan[1:4], [2602, 6, 2576])

    def test_example_contains_only_active_connected_workflow_nodes(self):
        self.assertTrue(all(node.get("mode", 0) == 0 for node in self.nodes.values()))
        self.assertNotIn("MiniMaxH3DirectorCS", {node["type"] for node in self.nodes.values()})
        self.assertNotIn("SamplerCustomAdvanced", {node["type"] for node in self.nodes.values()})
        referenced = {
            item["link"]
            for node in self.nodes.values()
            for item in node.get("inputs", ())
            if item.get("link") is not None
        }
        self.assertEqual(referenced, set(self.links))


if __name__ == "__main__":
    unittest.main()
