import { app } from "../../scripts/app.js";

// The frontend renders a ui text only on a node an extension opts in.
const NODE_TYPES = new Set(["LogitTagger", "LogitImageCheck", "LogitTextCheck", "LogitPromptCheck", "LogitImageClassify"]);
const FALLBACK_WIDGET = "tags_preview";

// The frontend's own text preview, the one Preview as Text uses. Older frontends lack it.
function textPreview() {
    return window.comfyAPI?.textPreviewWidgets;
}

function toText(text) {
    if (text == null) {
        return "";
    }
    return Array.isArray(text) ? text.filter((entry) => entry != null).join("\n\n") : String(text);
}

function addFallbackWidget(node) {
    const widgets = window.comfyAPI?.widgets?.ComfyWidgets;
    const widget = widgets?.STRING(node, FALLBACK_WIDGET, ["STRING", { multiline: true }], app).widget;

    if (!widget) {
        return;
    }
    // A preview never enters the saved workflow or the queued prompt.
    widget.serialize = false;
    widget.options.serialize = false;
    if (widget.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.6;
    }
}

function fillFallbackWidget(node, output) {
    const widget = node.widgets?.find((entry) => entry.name === FALLBACK_WIDGET);

    if (widget) {
        widget.value = toText(output?.text);
    }
}

function showPreview(node, output) {
    const preview = textPreview();

    if (preview?.updateTextPreviewWidgets) {
        preview.updateTextPreviewWidgets(node, output);
    } else {
        fillFallbackWidget(node, output);
    }
}

// The frontend keeps its own node lookups off window.comfyAPI.
function nodeByExecutionId(executionId) {
    const segments = String(executionId).split(":");
    let graph = app.rootGraph;
    let node = null;

    for (const segment of segments) {
        node = graph?.getNodeById(segment) ?? null;
        if (!node) {
            return null;
        }
        graph = node.subgraph;
    }
    return node;
}

// Frontend 1.53.6 hands onNodeOutputsUpdated node locator ids, "<subgraph uuid>:<local id>" for a node in a
// subgraph, and keeps every subgraph definition, nested ones included, in the root graph's subgraphs Map.
function nodeByOutputKey(key) {
    const parts = String(key).split(":");
    const subgraph = parts.length === 2 ? app.rootGraph?.subgraphs?.get?.(parts[0]) : null;

    if (subgraph) {
        return subgraph.getNodeById(parts[1]) ?? null;
    }
    return nodeByExecutionId(key);
}

app.registerExtension({
    name: "LogitTagger.TagPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData.name)) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const preview = textPreview();

            if (preview?.addTextPreviewWidgets) {
                preview.addTextPreviewWidgets(this);
            } else {
                addFallbackWidget(this);
            }
        };
        nodeType.prototype.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            showPreview(this, output);
        };
    },
    // A workflow tab switch and a queue history reopen restore outputs without onExecuted.
    onNodeOutputsUpdated(nodeOutputs) {
        for (const [key, output] of Object.entries(nodeOutputs)) {
            const node = nodeByOutputKey(key);

            if (!NODE_TYPES.has(node?.constructor?.comfyClass) && !NODE_TYPES.has(node?.type)) {
                continue;
            }
            showPreview(node, output);
        }
    },
});
