const { app } = window.comfyAPI.app;

const STYLE_ID = "hr-h3-prompt-timeline-style";
const COLORS = {
    shot: "#315f86",
    event: "#7c5c25",
    dialogue: "#356f4a",
    chunk: "#51476d",
    conflict: "#a63f3f",
};

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.hr-h3-timeline{box-sizing:border-box;height:430px;overflow:hidden;background:#17191d;color:#ddd;font:11px sans-serif;border:1px solid #444;border-radius:6px;display:flex;flex-direction:column}
.hr-h3-toolbar{display:flex;gap:8px;align-items:center;padding:6px;border-bottom:1px solid #3b3b3b;background:#202329}.hr-h3-toolbar button{background:#315f86;color:white;border:1px solid #5683a4;border-radius:4px;padding:4px 8px}.hr-h3-toolbar span{color:#9eb4c5}.hr-h3-conflicts{padding:5px 7px;min-height:18px;color:#ddd;background:#252128;border-bottom:1px solid #3b3b3b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.hr-h3-conflicts.bad{color:#ffb0a7;background:#3b2020}
.hr-h3-scroll{overflow:auto;flex:1}.hr-h3-grid{min-width:720px;position:relative;padding:4px 0 10px}.hr-h3-ruler,.hr-h3-row{display:grid;grid-template-columns:92px minmax(620px,1fr);min-height:34px}.hr-h3-label{position:sticky;left:0;z-index:5;background:#202329;border-right:1px solid #444;padding:8px 6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hr-h3-lane{position:relative;border-bottom:1px solid #30343a;background:repeating-linear-gradient(90deg,#1b1e23 0,#1b1e23 calc(10% - 1px),#30343a 10%)}
.hr-h3-ruler .hr-h3-lane{height:28px}.hr-h3-tick{position:absolute;top:0;bottom:0;border-left:1px solid #58606a;color:#aeb7c1;padding-left:3px;font-size:9px}.hr-h3-block{position:absolute;top:4px;height:25px;border:1px solid rgba(255,255,255,.32);border-radius:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:5px 7px;box-sizing:border-box;cursor:grab;color:#fff}.hr-h3-block.conflict{outline:2px solid #e05b50}.hr-h3-block.locked{cursor:default}.hr-h3-handle{position:absolute;top:0;bottom:0;width:7px;background:rgba(255,255,255,.2);cursor:ew-resize}.hr-h3-handle.left{left:0}.hr-h3-handle.right{right:0}.hr-h3-boundary{position:absolute;top:0;bottom:0;width:7px;margin-left:-3px;background:#7db2db;cursor:ew-resize;z-index:4}.hr-h3-chunkline{position:absolute;top:0;bottom:0;border-left:2px dashed #8d7eae;pointer-events:none;z-index:3}.hr-h3-help{padding:5px 7px;color:#95a0ac;border-top:1px solid #34383e}
`;
    document.head.append(style);
}

function planText(message) {
    const value = message?.text;
    if (Array.isArray(value)) return String(value[0] ?? "");
    return typeof value === "string" ? value : "";
}

function jsonWidget(node) {
    return node.widgets?.find(item => item.name === "edited_plan_json");
}

function chunkFrames(node) {
    const plan = parsePlan(node);
    const value = Number(plan?.chunk_frames ?? 124);
    return Number.isFinite(value) && value >= 22 ? Math.round(value) : 124;
}

function setWidgetText(node, text, render = true) {
    const widget = jsonWidget(node);
    if (!widget) return;
    widget.value = text;
    widget.callback?.(text);
    if (widget.inputEl) {
        widget.inputEl.value = text;
        widget.inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    node.graph?.setDirtyCanvas(true, true);
    if (render) node._hrPromptTimeline?.render();
}

function parsePlan(node) {
    const text = String(jsonWidget(node)?.value ?? "").trim();
    if (!text) return null;
    try {
        const value = JSON.parse(text);
        return value && typeof value === "object" && Array.isArray(value.shots) ? value : null;
    } catch {
        return null;
    }
}

function frameRange(item, shot) {
    return [Number(item.start_frame ?? shot.start_frame), Number(item.end_frame ?? shot.end_frame)];
}

function dialogueConflicts(plan, chunkSize) {
    const conflicts = [];
    const byChunk = new Map();
    const dialogues = [];
    for (const shot of plan.shots ?? []) {
        for (const item of shot.dialogues ?? []) {
            const [start, end] = frameRange(item, shot);
            dialogues.push({ item, start, end });
            const first = Math.floor(start / chunkSize);
            const last = Math.floor(Math.max(start, end - 1) / chunkSize);
            for (let chunk = first; chunk <= last; chunk++) {
                if (!byChunk.has(chunk)) byChunk.set(chunk, new Set());
                byChunk.get(chunk).add(String(item.speaker ?? ""));
            }
        }
    }
    for (const [chunk, speakers] of byChunk) {
        if (speakers.size > 1) conflicts.push(`Chunk ${chunk + 1} has ${speakers.size} speakers`);
    }
    dialogues.sort((a, b) => a.start - b.start);
    for (let i = 1; i < dialogues.length; i++) {
        if (dialogues[i].start < dialogues[i - 1].end) conflicts.push("Dialogue intervals overlap");
    }
    for (const shot of plan.shots ?? []) {
        for (const item of [...(shot.events ?? []), ...(shot.dialogues ?? [])]) {
            const [start, end] = frameRange(item, shot);
            if (start < shot.start_frame || end > shot.end_frame || end <= start) conflicts.push("Event/dialogue interval leaves its shot");
        }
    }
    return [...new Set(conflicts)];
}

function block(lane, item, shot, total, color, label, options, commit) {
    const [start, end] = frameRange(item, shot);
    const element = document.createElement("div");
    element.className = `hr-h3-block${options.locked ? " locked" : ""}${options.conflict ? " conflict" : ""}`;
    element.style.left = `${100 * start / total}%`;
    element.style.width = `${Math.max(.5, 100 * (end - start) / total)}%`;
    element.style.background = color;
    element.title = `${label}\nframes ${start}-${end}`;
    element.textContent = label;
    lane.appendChild(element);
    if (options.locked) return;
    for (const side of ["left", "right"]) {
        const handle = document.createElement("div");
        handle.className = `hr-h3-handle ${side}`;
        handle.dataset.side = side;
        element.appendChild(handle);
    }
    element.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault();
        const mode = event.target?.dataset?.side || "move";
        const laneRect = lane.getBoundingClientRect();
        const originX = event.clientX;
        const initialStart = start;
        const initialEnd = end;
        const minFrame = Number(shot.start_frame);
        const maxFrame = Number(shot.end_frame);
        const move = moveEvent => {
            const delta = Math.round((moveEvent.clientX - originX) * total / laneRect.width);
            let nextStart = initialStart;
            let nextEnd = initialEnd;
            if (mode === "move") {
                const length = initialEnd - initialStart;
                nextStart = Math.max(minFrame, Math.min(maxFrame - length, initialStart + delta));
                nextEnd = nextStart + length;
            } else if (mode === "left") {
                nextStart = Math.max(minFrame, Math.min(initialEnd - 1, initialStart + delta));
            } else {
                nextEnd = Math.min(maxFrame, Math.max(initialStart + 1, initialEnd + delta));
            }
            commit(nextStart, nextEnd, false);
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            commit(null, null, true);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up, { once: true });
    });
}

function createTimeline(node) {
    installStyle();
    const root = document.createElement("div");
    root.className = "hr-h3-timeline";
    root.innerHTML = `<div class="hr-h3-toolbar"><button type="button">Reload JSON</button><span>Qwen draft → review → validate → sampler</span></div><div class="hr-h3-conflicts">Run Compiler and Editor to load a plan.</div><div class="hr-h3-scroll"><div class="hr-h3-grid"></div></div><div class="hr-h3-help">Drag blocks to move; drag their edges to resize. Drag blue Shot boundaries. Dialogue text and speaker stay locked.</div>`;
    const grid = root.querySelector(".hr-h3-grid");
    const conflictBox = root.querySelector(".hr-h3-conflicts");
    root.querySelector("button").onclick = () => timeline.render();

    function save(plan, rerender = true) {
        setWidgetText(node, JSON.stringify(plan, null, 2), rerender);
    }

    function addRow(name) {
        const row = document.createElement("div");
        row.className = "hr-h3-row";
        const label = document.createElement("div");
        label.className = "hr-h3-label";
        label.textContent = name;
        const lane = document.createElement("div");
        lane.className = "hr-h3-lane";
        row.append(label, lane);
        grid.appendChild(row);
        return lane;
    }

    const timeline = {
        render() {
            const plan = parsePlan(node);
            grid.replaceChildren();
            if (!plan || !Number(plan.total_frames)) {
                conflictBox.className = "hr-h3-conflicts bad";
                conflictBox.textContent = "No valid plan JSON. Run Compiler → Editor or paste JSON.";
                return;
            }
            const total = Number(plan.total_frames);
            const fps = Number(plan.fps || 24);
            const chunkSize = chunkFrames(node);
            const conflicts = dialogueConflicts(plan, chunkSize);
            conflictBox.className = `hr-h3-conflicts${conflicts.length ? " bad" : ""}`;
            conflictBox.textContent = conflicts.length ? conflicts.join(" · ") : "No timeline conflicts detected.";

            const ruler = document.createElement("div");
            ruler.className = "hr-h3-ruler";
            const title = document.createElement("div"); title.className = "hr-h3-label"; title.textContent = `Time @ ${fps}fps`;
            const rulerLane = document.createElement("div"); rulerLane.className = "hr-h3-lane";
            for (let index = 0; index <= 10; index++) {
                const tick = document.createElement("div"); tick.className = "hr-h3-tick"; tick.style.left = `${index * 10}%`;
                tick.textContent = `${(total * index / 10 / fps).toFixed(1)}s`;
                rulerLane.appendChild(tick);
            }
            ruler.append(title, rulerLane); grid.appendChild(ruler);

            const chunkLane = addRow(`Chunks (${chunkSize}f)`);
            for (let start = 0, index = 1; start < total; start += chunkSize, index++) {
                const item = { start_frame: start, end_frame: Math.min(total, start + chunkSize) };
                block(chunkLane, item, item, total, COLORS.chunk, `C${index}`, { locked: true }, () => {});
            }

            const shotLane = addRow("Shots");
            (plan.shots ?? []).forEach((shot, index) => {
                block(shotLane, shot, shot, total, COLORS.shot, `Shot ${index + 1}`, { locked: true }, () => {});
                if (index > 0) {
                    const boundary = document.createElement("div");
                    boundary.className = "hr-h3-boundary";
                    boundary.style.left = `${100 * shot.start_frame / total}%`;
                    boundary.title = `Drag Shot ${index}/${index + 1} boundary`;
                    shotLane.appendChild(boundary);
                    boundary.addEventListener("pointerdown", event => {
                        event.preventDefault();
                        const rect = shotLane.getBoundingClientRect();
                        const previous = plan.shots[index - 1];
                        const current = plan.shots[index];
                        const move = moveEvent => {
                            const frame = Math.max(previous.start_frame + 1, Math.min(current.end_frame - 1, Math.round((moveEvent.clientX - rect.left) * total / rect.width)));
                            previous.end_frame = frame;
                            current.start_frame = frame;
                            save(plan, false);
                            timeline.render();
                        };
                        const up = () => { window.removeEventListener("pointermove", move); save(plan); };
                        window.addEventListener("pointermove", move);
                        window.addEventListener("pointerup", up, { once: true });
                    });
                }
            });

            const eventLane = addRow("Actions");
            for (const shot of plan.shots ?? []) {
                for (const event of shot.events ?? []) {
                    block(eventLane, event, shot, total, COLORS.event, `${event.id || "event"}: ${event.action || ""}`, {}, (start, end, final) => {
                        if (start !== null) { event.start_frame = start; event.end_frame = end; }
                        save(plan, final);
                    });
                }
            }

            const dialogueLane = addRow("Dialogue");
            const conflictChunks = new Set();
            const chunkSpeakers = new Map();
            for (const shot of plan.shots ?? []) for (const dialogue of shot.dialogues ?? []) {
                const [start, end] = frameRange(dialogue, shot);
                for (let c = Math.floor(start / chunkSize); c <= Math.floor(Math.max(start, end - 1) / chunkSize); c++) {
                    if (!chunkSpeakers.has(c)) chunkSpeakers.set(c, new Set());
                    chunkSpeakers.get(c).add(dialogue.speaker);
                }
            }
            for (const [chunk, speakers] of chunkSpeakers) if (speakers.size > 1) conflictChunks.add(chunk);
            for (const shot of plan.shots ?? []) {
                for (const dialogue of shot.dialogues ?? []) {
                    const [start, end] = frameRange(dialogue, shot);
                    let conflict = false;
                    for (let c = Math.floor(start / chunkSize); c <= Math.floor(Math.max(start, end - 1) / chunkSize); c++) conflict ||= conflictChunks.has(c);
                    block(dialogueLane, dialogue, shot, total, COLORS.dialogue, `${dialogue.speaker_id || "S?"} ${dialogue.speaker || ""}: ${dialogue.text || ""}`, { conflict }, (nextStart, nextEnd, final) => {
                        if (nextStart !== null) { dialogue.start_frame = nextStart; dialogue.end_frame = nextEnd; }
                        save(plan, final);
                    });
                }
            }

            for (let start = chunkSize; start < total; start += chunkSize) {
                for (const lane of grid.querySelectorAll(".hr-h3-lane")) {
                    if (lane === rulerLane) continue;
                    const line = document.createElement("div"); line.className = "hr-h3-chunkline"; line.style.left = `${100 * start / total}%`; lane.appendChild(line);
                }
            }
        },
    };
    node.addDOMWidget("prompt_timeline", "div", root, { serialize: false });
    node.setSize([Math.max(node.size[0], 760), Math.max(node.size[1], 720)]);
    node._hrPromptTimeline = timeline;
    return timeline;
}

function updateEditor(node, text) {
    if (!text) return;
    setWidgetText(node, text, false);
    node.setSize([Math.max(node.size[0], 760), Math.max(node.size[1], 720)]);
    node._hrPromptTimeline?.render();
}

app.registerExtension({
    name: "hr-endless-sampler.prompt-plan-editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HRH3PromptPlanEditor") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const timeline = createTimeline(this);
            const widget = jsonWidget(this);
            widget?.inputEl?.addEventListener("input", () => timeline.render());
            timeline.render();
        };
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            original?.apply(this, arguments);
            updateEditor(this, planText(message));
        };
    },
});
