const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const style = `
.hr-retake {width:100%;height:100%;box-sizing:border-box;padding:10px;overflow:auto;background:#181a1f;color:#ddd;font:12px sans-serif}
.hr-retake .top {display:flex;gap:8px;align-items:center;margin-bottom:10px;position:sticky;top:0;background:#181a1f;padding:4px 0;z-index:2}
.hr-retake button {background:#294963;color:#fff;border:1px solid #5683a4;border-radius:5px;padding:6px 10px;cursor:pointer}
.hr-retake .status {color:#9bb5c8}.hr-retake .grid {display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.hr-retake .card {background:#22262d;border:1px solid #404854;border-radius:8px;padding:9px}.hr-retake .bad {border-color:#985a4d}
.hr-retake .title {font-weight:bold;color:#8fc7e8;margin-bottom:7px}.hr-retake .images {display:flex;gap:5px;overflow:auto;margin-bottom:8px}
.hr-retake img {height:92px;max-width:160px;object-fit:contain;background:#111;border-radius:4px}.hr-retake details {margin-top:6px}
.hr-retake pre {white-space:pre-wrap;max-height:190px;overflow:auto;background:#16181c;padding:7px;border-radius:5px;color:#d6d9df}
`;

function installStyle() {
    if (document.getElementById("hr-retake-style")) return;
    const element = document.createElement("style");
    element.id = "hr-retake-style";
    element.textContent = style;
    document.head.appendChild(element);
}

function formatTime(frame, fps) {
    const seconds = Number(frame) / (Number(fps) || 24);
    return Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : "?";
}

function promptDetails(label, text) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = label;
    const pre = document.createElement("pre");
    pre.textContent = String(text || "（无）");
    details.append(summary, pre);
    return details;
}

function createDirector(node) {
    installStyle();
    const root = document.createElement("div");
    root.className = "hr-retake";
    const top = document.createElement("div");
    top.className = "top";
    const refresh = document.createElement("button");
    refresh.textContent = "刷新最后运行缓存";
    const status = document.createElement("span");
    status.className = "status";
    top.append(refresh, status);
    const grid = document.createElement("div");
    grid.className = "grid";
    root.append(top, grid);

    async function load() {
        status.textContent = "读取中…";
        grid.replaceChildren();
        try {
            const response = await api.fetchApi("/hr_endless_sampler_retake/cache");
            const data = await response.json();
            if (!data.available) {
                status.textContent = data.reason || "没有可用缓存";
                return;
            }
            status.textContent = `状态：${data.status} · 已完成 ${data.completed_chunks} 段 · 缓存格式 v${data.format}${data.compatible ? "" : "（不兼容）"}`;
            const fps = data.fps || 24;
            for (const chunk of data.chunks || []) {
                const card = document.createElement("section");
                card.className = `card${chunk.complete ? "" : " bad"}`;
                const title = document.createElement("div");
                title.className = "title";
                title.textContent = `Chunk ${chunk.chunk} · ${formatTime(chunk.frame_start, fps)}–${formatTime(chunk.frame_end, fps)} · ${chunk.complete ? "缓存完整" : "缓存不完整"}`;
                const images = document.createElement("div");
                images.className = "images";
                for (const path of chunk.observation_images || []) {
                    const image = document.createElement("img");
                    image.src = api.apiURL(`/hr_endless_sampler_retake/asset?path=${encodeURIComponent(path)}`);
                    image.loading = "lazy";
                    images.appendChild(image);
                }
                if (!images.childElementCount) images.textContent = "没有观察图片";
                card.append(title, images,
                    promptDetails("最终 H3 提示词", chunk.effective_h3_prompt),
                    promptDetails("原始完整提示词", chunk.source_prompt));
                if (chunk.error) card.append(promptDetails("缓存错误", chunk.error));
                grid.appendChild(card);
            }
        } catch (error) {
            status.textContent = `读取失败：${error.message || error}`;
        }
    }
    refresh.onclick = load;
    node.addDOMWidget("retake_cache_view", "div", root, {serialize:false});
    node.setSize([Math.max(node.size[0], 760), Math.max(node.size[1], 520)]);
    load();
}

app.registerExtension({
    name: "hr-endless-sampler.retake-director",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HREndlessSegmentRetakeDirector") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            original?.apply(this, arguments);
            createDirector(this);
        };
    },
});
