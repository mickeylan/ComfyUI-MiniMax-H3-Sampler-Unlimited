const { app } = window.comfyAPI.app;

function planText(message) {
    const value = message?.text;
    if (Array.isArray(value)) return String(value[0] ?? "");
    return typeof value === "string" ? value : "";
}

function updateEditor(node, text) {
    if (!text) return;
    const widget = node.widgets?.find(item => item.name === "edited_plan_json");
    if (!widget) return;
    widget.value = text;
    if (widget.inputEl) {
        widget.inputEl.value = text;
        widget.inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    node.setSize([Math.max(node.size[0], 620), Math.max(node.size[1], 520)]);
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "hr-endless-sampler.prompt-plan-editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HRH3PromptPlanEditor") return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            original?.apply(this, arguments);
            updateEditor(this, planText(message));
        };
    },
});
