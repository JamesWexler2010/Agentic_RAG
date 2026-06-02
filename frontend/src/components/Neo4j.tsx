import { useEffect, useRef, useState } from "react";
import NeoVis from "neovis.js";
import { 
  Loader2, AlertCircle, RefreshCw, Maximize2, ExternalLink, 
  Play, Sparkles, MessageSquareQuote 
} from "lucide-react";
import { Button } from "./ui/button";

interface GraphViewProps {
  fileId?: string;
  // ✨ 新增:由父组件(PDFPanel ← App)下发的"本次问答涉及子图"的 Cypher。
  //    收到非空字符串时,GraphView 会自动切换到这段 Cypher 并重新渲染。
  externalCypher?: string;
}

type Status = "loading" | "ready" | "empty" | "error";

// ⚠️ 改成你自己的 Neo4j 密码
const NEO4J_CONFIG = {
  serverUrl: "bolt://localhost:7687",
  serverUser: "neo4j",
  serverPassword: "20472036",
};

const DEFAULT_CYPHER = "MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 200";

// 用一个特殊 label 标记"来自问答"的预设项,显示时高亮
const QA_PRESET_LABEL = "📍本次问答";

const PRESETS = [
  {
    label: "全部结构",
    query: "MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 200",
    desc: "展示全部节点与关系",
  },
  {
    label: "章节树",
    query: "MATCH (n:Section)-[r:HAS_CHILD]->(m:Section) RETURN n, r, m LIMIT 100",
    desc: "只看章节父子层级",
  },
  {
    label: "章节+正文",
    query: "MATCH (s:Section)-[r:HAS_CHUNK]->(c:Chunk) RETURN s, r, c LIMIT 150",
    desc: "章节及其正文",
  },
  {
    label: "所有表格",
    query: "MATCH (n)-[r:CONTAINS_TABLE]->(t:Table) OPTIONAL MATCH (t)-[r2:HAS_ROW]->(row) RETURN n, r, t, r2, row LIMIT 150",
    desc: "表格及其所在节点和行",
  },
  {
    label: "表格行链",
    query: "MATCH (t:Table)-[r:HAS_ROW]->(row:TableRow) OPTIONAL MATCH (row)-[r2:NEXT_ROW]->(next) RETURN t, r, row, r2, next LIMIT 200",
    desc: "表格行顺序链",
  },
  {
    label: "所有图片",
    query: "MATCH (n)-[r:CONTAINS_IMAGE]->(i:Image) RETURN n, r, i LIMIT 100",
    desc: "图片及其上下文",
  },
  {
    label: "顶层章节",
    query: "MATCH (n:Section) WHERE n.depth <= 2 RETURN n LIMIT 50",
    desc: "仅顶层(1-2级)章节",
  },
];

export function GraphView({ fileId, externalCypher }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const vizRef = useRef<any>(null);

  const [status, setStatus] = useState<Status>("loading");
  const [errorMsg, setErrorMsg] = useState("");

  const [cypherInput, setCypherInput] = useState(DEFAULT_CYPHER);
  const [currentQuery, setCurrentQuery] = useState(DEFAULT_CYPHER);

  // ✨ 记录"本次问答 Cypher"(用于在预设按钮里高亮"本次问答"项)
  const [qaCypher, setQaCypher] = useState<string>("");
  // ✨ 记录本次问答涉及的节点数(用于按钮提示)
  const [qaNodeCount, setQaNodeCount] = useState<number>(0);

  const containerId = `neovis-${fileId || "default"}`;

  // ✨ 监听 externalCypher 变化:有新值就自动切换并显示
  // 这是方案 a:自动覆盖用户的当前查询
  useEffect(() => {
    if (!externalCypher) return;
    if (externalCypher === qaCypher) return;  // 同样的 cypher 不重复触发

    console.log("[GraphView] 收到外部 Cypher,自动切换视图");
    setQaCypher(externalCypher);
    // 估算节点数(从 Cypher 中读 IN [...] 列表的总长度近似)
    // 简单做法:每个 entity_id IN [...] 数一下逗号 + 1
    try {
      const matches = externalCypher.match(/IN\s*\[([^\]]+)\]/g) || [];
      let total = 0;
      for (const m of matches) {
        // 数引号对(每个 id 两个引号)
        const quoteCount = (m.match(/"/g) || []).length;
        total += Math.floor(quoteCount / 2);
      }
      setQaNodeCount(total);
    } catch {
      setQaNodeCount(0);
    }

    // 同步到输入框 + 触发渲染
    setCypherInput(externalCypher);
    setCurrentQuery(externalCypher);
  }, [externalCypher, qaCypher]);

  const runViz = (query: string) => {
    if (!containerRef.current) return;

    if (vizRef.current?.clearNetwork) {
      try {
        vizRef.current.clearNetwork();
      } catch {}
    }

    setStatus("loading");
    setErrorMsg("");

    const config = {
      containerId,
      neo4j: {
        serverUrl: NEO4J_CONFIG.serverUrl,
        serverUser: NEO4J_CONFIG.serverUser,
        serverPassword: NEO4J_CONFIG.serverPassword,
      },

      labels: {
        Section: { label: "entity_name" },
        Chunk:   { label: "entity_name" },
        Table:   { label: "entity_name" },
        TableRow:{ label: "display_name" },
        Image:   { label: "entity_name" },
      },

      relationships: {
        HAS_CHILD: {}, HAS_CHUNK: {}, CONTAINS_IMAGE: {},
        CONTAINS_TABLE: {}, HAS_ROW: {}, NEXT_ROW: {},
      },

      initialCypher: query,

      visConfig: {
        nodes: {
          shape: "dot",
          size: 22,
          font: {
            size: 13, color: "#e2e8f0", strokeWidth: 3,
            strokeColor: "#0f172a", face: "Arial",
          },
          borderWidth: 2,
        },
        groups: {
          Section: {
            color: {
              background: "#3b82f6", border: "#60a5fa",
              highlight: { background: "#60a5fa", border: "#93c5fd" },
            },
            shape: "dot",
          },
          Chunk: {
            color: {
              background: "#10b981", border: "#34d399",
              highlight: { background: "#34d399", border: "#6ee7b7" },
            },
            shape: "dot",
          },
          Table: {
            color: {
              background: "#f59e0b", border: "#fbbf24",
              highlight: { background: "#fbbf24", border: "#fcd34d" },
            },
            shape: "dot",
          },
          TableRow: {
            color: {
              background: "#a855f7", border: "#c084fc",
              highlight: { background: "#c084fc", border: "#d8b4fe" },
            },
            shape: "dot",
            size: 14,
          },
          Image: {
            color: {
              background: "#ec4899", border: "#f472b6",
              highlight: { background: "#f472b6", border: "#f9a8d4" },
            },
            shape: "dot",
          },
        },
        edges: {
          arrows: { to: { enabled: true, scaleFactor: 0.5 } },
          smooth: { type: "continuous" },
          font: {
            size: 10, color: "#cbd5e1", strokeWidth: 3,
            strokeColor: "#0f172a", align: "middle",
          },
          color: { color: "#64748b", highlight: "#3b82f6" },
          length: 180,
        },
        physics: {
          enabled: true,
          barnesHut: {
            gravitationalConstant: -10000, springLength: 180,
            springConstant: 0.04, avoidOverlap: 0.5,
          },
          stabilization: { iterations: 250 },
        },
        interaction: {
          hover: true, tooltipDelay: 200,
          zoomView: true, dragView: true, hideEdgesOnDrag: false,
        },
      },
    };

    try {
      const viz = new (NeoVis as any)(config);

      viz.registerOnEvent?.("completed", (e: any) => {
        const count = e?.recordCount ?? 0;
        console.log("[Neovis] completed, records:", count);

        try {
          const network = vizRef.current?.network;
          if (network) {
            const nodesData = network.body.data.nodes;
            const edgesData = network.body.data.edges;

            const nodeIds = nodesData.getIds();
            const nodeUpdates: any[] = [];
            nodeIds.forEach((id: any) => {
              const node = nodesData.get(id);
              const raw = node?.raw?.properties;
              const labels = node?.raw?.labels || [];
              if (labels.includes("TableRow") && raw) {
                const parent = raw.parent_entity_name || "未知表格";
                const idx = raw.row_index !== undefined ? raw.row_index : "?";
                const display = `${parent} · 行${idx}`;
                if (!node.label || node.label === "undefined") {
                  nodeUpdates.push({ id, label: display });
                }
              }
            });
            if (nodeUpdates.length > 0) nodesData.update(nodeUpdates);

            // 关系上色
            const edgeIds = edgesData.getIds();
            const edgeUpdates: any[] = [];
            edgeIds.forEach((id: any) => {
              const edge = edgesData.get(id);
              const relType = edge?.raw?.type || edge?.label;
              const colorMap: Record<string, { color: string; dashes?: boolean }> = {
                HAS_CHILD:       { color: "#60a5fa" },
                HAS_CHUNK:       { color: "#34d399" },
                CONTAINS_TABLE:  { color: "#fbbf24" },
                CONTAINS_IMAGE:  { color: "#f472b6" },
                HAS_ROW:         { color: "#c084fc" },
                NEXT_ROW:        { color: "#94a3b8", dashes: true },
              };
              const style = colorMap[relType];
              if (style) {
                edgeUpdates.push({
                  id,
                  label: relType,
                  color: { color: style.color, highlight: style.color },
                  dashes: style.dashes || false,
                });
              }
            });
            if (edgeUpdates.length > 0) {
              edgesData.update(edgeUpdates);
            }
          }
        } catch (postErr) {
          console.warn("[Neovis] post-process failed:", postErr);
        }

        if (count === 0) setStatus("empty");
        else setStatus("ready");
      });

      viz.registerOnEvent?.("error", (e: any) => {
        console.error("[Neovis error]", e);
        setErrorMsg(
          e?.error?.message || "查询失败,请检查 Cypher 语法或 Neo4j 连接"
        );
        setStatus("error");
      });

      viz.render();
      vizRef.current = viz;
    } catch (err: any) {
      console.error("[Neovis init error]", err);
      setErrorMsg(err?.message || "初始化失败");
      setStatus("error");
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => runViz(currentQuery), 100);
    return () => {
      clearTimeout(timer);
      if (vizRef.current?.clearNetwork) {
        try {
          vizRef.current.clearNetwork();
        } catch {}
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileId, currentQuery]);

  const handleExecute = () => {
    const trimmed = cypherInput.trim();
    if (!trimmed) return;
    setCurrentQuery(trimmed);
  };

  const handlePreset = (query: string) => {
    setCypherInput(query);
    setCurrentQuery(query);
  };

  const handleRefresh = () => runViz(currentQuery);

  const handleFullscreen = () => {
    containerRef.current?.requestFullscreen?.();
  };

  const openNeo4jBrowser = () => {
    window.open("http://localhost:7474", "_blank");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleExecute();
    }
  };

  // ✨ "本次问答"按钮是否高亮:当前查询 === qaCypher
  const isQaActive = !!qaCypher && currentQuery === qaCypher;

  return (
    <div className="w-full h-full flex flex-col relative">
      {/* 顶部工具栏 */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/40 bg-card/20">
        <span className="text-xs text-muted-foreground flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5" />
          知识图谱
        </span>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={handleRefresh} className="h-7 px-2" title="刷新当前查询">
            <RefreshCw className="w-3.5 h-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={handleFullscreen} className="h-7 px-2" title="全屏">
            <Maximize2 className="w-3.5 h-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={openNeo4jBrowser} className="h-7 px-2 text-xs gap-1" title="在 Neo4j Browser 中打开">
            <ExternalLink className="w-3.5 h-3.5" />
            Browser
          </Button>
        </div>
      </div>

      {/* Cypher 查询输入区 */}
      <div className="px-3 py-2 border-b border-border/40 bg-card/10 space-y-2">
        <div className="flex gap-2">
          <textarea
            value={cypherInput}
            onChange={(e) => setCypherInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入 Cypher,Ctrl+Enter 执行"
            className="flex-1 px-3 py-2 text-xs font-mono bg-slate-900/60 border border-border/40 rounded-md resize-none focus:outline-none focus:border-blue-500/50 text-foreground placeholder:text-muted-foreground/50"
            rows={2}
            spellCheck={false}
          />
          <Button
            onClick={handleExecute}
            size="sm"
            className="h-auto px-4 bg-blue-500/20 border border-blue-500/40 text-blue-400 hover:bg-blue-500/30 gap-1.5"
            title="执行 (Ctrl+Enter)"
          >
            <Play className="w-3.5 h-3.5" />
            执行
          </Button>
        </div>

        {/* 预设按钮 */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-muted-foreground/60">视图:</span>

          {/* ✨ "本次问答"按钮:仅当用户已经在图谱模式问过问题后显示 */}
          {qaCypher && (
            <button
              onClick={() => handlePreset(qaCypher)}
              className={`text-[10px] px-2 py-0.5 rounded-full border transition-all inline-flex items-center gap-1 ${
                isQaActive
                  ? "bg-amber-500/20 border-amber-500/40 text-amber-300"
                  : "bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20"
              }`}
              title={`点击查看本次问答涉及的子图(约 ${qaNodeCount} 个节点)`}
            >
              <MessageSquareQuote className="w-3 h-3" />
              {QA_PRESET_LABEL}
              <span className="text-[9px] opacity-70">({qaNodeCount})</span>
            </button>
          )}

          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => handlePreset(p.query)}
              className={`text-[10px] px-2 py-0.5 rounded-full border transition-all ${
                currentQuery === p.query
                  ? "bg-blue-500/20 border-blue-500/40 text-blue-400"
                  : "bg-secondary/40 border-border/40 text-muted-foreground hover:bg-secondary/60"
              }`}
              title={p.desc}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* 关系图例 */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-[10px] text-muted-foreground/60">关系:</span>
          <EdgeLegend color="#60a5fa" label="HAS_CHILD" />
          <EdgeLegend color="#34d399" label="HAS_CHUNK" />
          <EdgeLegend color="#fbbf24" label="CONTAINS_TABLE" />
          <EdgeLegend color="#f472b6" label="CONTAINS_IMAGE" />
          <EdgeLegend color="#c084fc" label="HAS_ROW" />
          <EdgeLegend color="#94a3b8" label="NEXT_ROW" dashed />
        </div>
      </div>

      {/* 图容器 */}
      <div className="flex-1 relative bg-slate-900/60">
        <div id={containerId} ref={containerRef} className="absolute inset-0" />

        {status === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-2 text-muted-foreground">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span className="text-xs">执行查询中...</span>
            </div>
          </div>
        )}

        {status === "empty" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center space-y-2 max-w-xs">
              <AlertCircle className="w-10 h-10 text-muted-foreground/60 mx-auto" />
              <p className="text-sm text-muted-foreground">查询无结果</p>
              <p className="text-xs text-muted-foreground/60">
                尝试换一个预设或修改 Cypher
              </p>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="text-center space-y-2 max-w-sm">
              <AlertCircle className="w-10 h-10 text-red-500/80 mx-auto" />
              <p className="text-sm text-foreground">查询出错</p>
              <p className="text-xs text-muted-foreground/80 break-words font-mono bg-slate-900/60 p-2 rounded border border-border/40">
                {errorMsg}
              </p>
              <div className="flex gap-2 justify-center mt-3">
                <Button variant="outline" size="sm" onClick={handleRefresh}>
                  重试
                </Button>
                <Button variant="outline" size="sm" onClick={openNeo4jBrowser}>
                  打开 Browser
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EdgeLegend({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <span
        style={{
          width: 14,
          height: 2,
          backgroundColor: dashed ? "transparent" : color,
          borderTop: dashed ? `2px dashed ${color}` : "none",
          display: "inline-block",
        }}
      />
      <span className="text-[10px] text-muted-foreground font-mono">{label}</span>
    </div>
  );
}