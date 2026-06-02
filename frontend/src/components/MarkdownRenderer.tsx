import React, { useMemo, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import { defaultSchema } from "hast-util-sanitize";
import { FileText, ExternalLink, ImageIcon, ChevronDown, ChevronUp } from "lucide-react";
import { getCitationChunk } from "../services/api";

/** 可改成从 .env 读取 */
const API_BASE = "http://localhost:8001/api/v1";
const API_HOST = String(API_BASE).replace(/\/api\/v\d+$/, ""); // http://localhost:8001

// snippet 超过这个字符数时默认折叠
const SNIPPET_COLLAPSE_THRESHOLD = 150;

// ⭐ 允许 <img> 的自定义 schema
const sanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "img"],
  attributes: {
    ...(defaultSchema.attributes || {}),
    "*": [...((defaultSchema.attributes && defaultSchema.attributes["*"]) || []), "className"],
    img: [
      "src",
      "alt",
      "title",
      "loading",
      "width",
      "height",
      "className",
    ],
    a: [
      ...((defaultSchema.attributes && defaultSchema.attributes["a"]) || []),
      "target",
      "rel",
    ],
  },
  protocols: {
    ...(defaultSchema.protocols || {}),
    src: ["http", "https", "data", "blob"],
    href: ["http", "https", "mailto", "tel"],
  },
};

/** /api/v1/... 相对路径 -> 绝对地址 */
function toAbsoluteApiUrl(src: string) {
  if (!src) return "";
  if (src.startsWith("http://") || src.startsWith("https://")) return src;
  if (src.startsWith("/api/")) return `${API_HOST}${src}`;
  return src;
}

/** 代码块（带复制） */
function Code(props: any) {
  const { inline, className, children } = props;
  const language = (className || "").replace("language-", "") || "code";
  if (inline) {
    return <code className="bg-muted/50 px-1.5 py-0.5 rounded text-sm">{children}</code>;
  }
  return (
    <div className="my-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">{language}</span>
        <button
          className="text-xs px-2 py-1 rounded bg-white/10 hover:bg-white/20"
          onClick={() => navigator.clipboard.writeText(String(children))}
        >
          Copy
        </button>
      </div>
      <pre className="text-sm overflow-x-auto bg-slate-900/80 p-3 rounded border border-slate-700/50">
        <code className="text-slate-200">{children}</code>
      </pre>
    </div>
  );
}

/** 图片：忽略本地相对路径，只渲染可访问的 API/HTTP 图片 */
function Img(props: React.ImgHTMLAttributes<HTMLImageElement>) {
  const fixedSrc = useMemo(() => {
    const src = String(props.src || "");
    if (!src) return "";
    if (src.startsWith("./images/") || src.startsWith("images/")) return "";
    return toAbsoluteApiUrl(src);
  }, [props.src]);

  const [err, setErr] = useState(false);
  if (!fixedSrc || err) return null;

  return (
    <img
      {...props}
      src={fixedSrc}
      onError={() => setErr(true)}
      className={"max-w-full h-auto rounded-lg border border-border/30 shadow-sm " + (props.className ?? "")}
      loading="lazy"
    />
  );
}

// ✨ 图片对象类型
type ImageItem = {
  alt: string;
  url: string;
};

/** 引用卡片底部的图片画廊 */
function ImageGallery({
  images = [],
  tableImgPath = [],
}: {
  images?: ImageItem[];
  tableImgPath?: string[];
}) {
  const merged = useMemo(() => {
    const seen = new Set<string>();
    const result: ImageItem[] = [];
    for (const img of images) {
      if (img?.url && !seen.has(img.url)) {
        seen.add(img.url);
        result.push(img);
      }
    }
    for (const url of tableImgPath) {
      if (url && !seen.has(url)) {
        seen.add(url);
        result.push({ alt: "表格图片", url });
      }
    }
    return result;
  }, [images, tableImgPath]);

  if (merged.length === 0) return null;

  return (
    <div className="mt-3">
      <div className="flex items-center gap-1.5 mb-2 text-xs text-muted-foreground">
        <ImageIcon className="w-3 h-3" />
        <span>相关图片 ({merged.length})</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {merged.map((img, i) => (
          <a
            key={i}
            href={img.url}
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg overflow-hidden border border-border/30 hover:border-primary/40 transition-colors bg-muted/10"
            title={img.alt}
          >
            <img
              src={img.url}
              alt={img.alt}
              loading="lazy"
              className="w-full h-32 object-cover hover:scale-105 transition-transform"
            />
          </a>
        ))}
      </div>
    </div>
  );
}

/**
 * ✨ snippet 折叠组件
 *   - 短文本(< 阈值)直接全展开
 *   - 长文本默认折叠到 2 行,带"展开 ▾ / 收起 ▴"按钮
 */
function CollapsibleSnippet({ snippet }: { snippet: string }) {
  // 是否值得折叠(短文本就别折了)
  const shouldCollapse = snippet.length > SNIPPET_COLLAPSE_THRESHOLD;
  const [expanded, setExpanded] = useState(false);

  // 文本太短直接全展开,无折叠按钮
  if (!shouldCollapse) {
    return (
      <div className="text-sm text-foreground leading-relaxed prose prose-invert prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
          components={mdComponents}
        >
          {snippet}
        </ReactMarkdown>
      </div>
    );
  }

  return (
    <div className="text-sm text-foreground leading-relaxed">
      {/* Markdown 渲染区域:折叠态用 CSS line-clamp 限制 2 行,展开态滚动 */}
      <div
        className={
          "prose prose-invert prose-sm max-w-none transition-all " +
          (expanded
            ? "max-h-[400px] overflow-y-auto"
            // line-clamp-2:超出2行用省略号截断
            : "max-h-[3em] overflow-hidden text-muted-foreground/90 [&_*]:!my-0 [&_h1]:!text-sm [&_h2]:!text-sm [&_h3]:!text-sm [&_h4]:!text-sm [&_h5]:!text-sm [&_h6]:!text-sm")
        }
        style={
          expanded
            ? undefined
            : {
                // 多浏览器兼容的 line-clamp
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
              }
        }
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
          components={mdComponents}
        >
          {snippet}
        </ReactMarkdown>
      </div>

      {/* 折叠/展开按钮 */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="mt-1.5 inline-flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors"
      >
        {expanded ? (
          <>
            <ChevronUp className="w-3 h-3" />
            收起
          </>
        ) : (
          <>
            <ChevronDown className="w-3 h-3" />
            展开全文 ({snippet.length} 字)
          </>
        )}
      </button>
    </div>
  );
}

// Markdown 组件配置(在 CollapsibleSnippet 内复用)
const mdComponents = {
  img: Img,
  code: Code,
  p: (p: any) => <p {...p} className="my-1" />,
  h1: (p: any) => <h1 {...p} className="text-lg font-medium mt-2 mb-1" />,
  h2: (p: any) => <h2 {...p} className="text-base font-medium mt-2 mb-1" />,
  h3: (p: any) => <h3 {...p} className="text-sm font-medium mt-2 mb-1" />,
  h4: (p: any) => <h4 {...p} className="text-sm font-medium mt-1.5 mb-0.5" />,
  h5: (p: any) => <h5 {...p} className="text-sm font-medium mt-1 mb-0.5" />,
  h6: (p: any) => <h6 {...p} className="text-xs font-medium mt-1 mb-0.5" />,
  table: (p: any) => <table {...p} className="w-full border-collapse border border-border/30 rounded-lg overflow-hidden my-2" />,
  thead: (p: any) => <thead {...p} className="bg-muted/30" />,
  th: (p: any) => <th {...p} className="px-2 py-1 border border-border/30 text-left text-xs font-medium" />,
  td: (p: any) => <td {...p} className="px-2 py-1 border border-border/30 text-xs" />,
};

/** 懒加载 citation 详情,snippet 用 Markdown 渲染 + 折叠 */
function ReferenceCard({
  citationId,
  index,
  fallbackImages = [],
  fallbackTableImgPath = [],
}: {
  citationId: string;
  index: number;
  fallbackImages?: ImageItem[];
  fallbackTableImgPath?: string[];
}) {
  const [loading, setLoading] = useState(false);
  const [snippet, setSnippet] = useState<string>("");
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [chunkImages, setChunkImages] = useState<ImageItem[]>([]);
  const [chunkTableImgPath, setChunkTableImgPath] = useState<string[]>([]);
  const [loadedFromApi, setLoadedFromApi] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        const chunk: any = await getCitationChunk(citationId);
        if (!mounted) return;
        setSnippet(chunk?.snippet || "");
        const urls = chunk?.previewUrls?.length
          ? chunk.previewUrls.map(toAbsoluteApiUrl)
          : chunk?.previewUrl
          ? [toAbsoluteApiUrl(chunk.previewUrl)]
          : [];
        setPreviewUrls(urls);
        if (Array.isArray(chunk?.images)) setChunkImages(chunk.images);
        if (Array.isArray(chunk?.table_img_path)) setChunkTableImgPath(chunk.table_img_path);
        setLoadedFromApi(true);
      } catch (e) {
        console.warn("[ReferenceCard] getCitationChunk failed, fallback to SSE data:", e);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, [citationId]);

  // 优先用 API 加载的数据,失败时用 SSE 推送的 fallback
  const displayImages = loadedFromApi ? chunkImages : fallbackImages;
  const displayTableImgPath = loadedFromApi ? chunkTableImgPath : fallbackTableImgPath;

  return (
    <div className="bg-muted/20 rounded-lg p-3 border border-border/30">
      <div className="flex items-start gap-3">
        <span className="inline-flex items-center justify-center w-6 h-6 text-xs font-medium bg-primary/20 text-primary rounded-full shrink-0">
          {index + 1}
        </span>
        <div className="flex-1 min-w-0">
          {/* ✨ snippet 折叠显示 */}
          {loading ? (
            <div className="text-sm text-muted-foreground">加载中…</div>
          ) : snippet ? (
            <CollapsibleSnippet snippet={snippet} />
          ) : (
            <div className="text-sm text-muted-foreground">（无文本片段）</div>
          )}

          {/* 图片画廊 */}
          <ImageGallery
            images={displayImages}
            tableImgPath={displayTableImgPath}
          />

          {/* 多页按钮 */}
          {previewUrls.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {previewUrls.map((url, i) => (
                <button
                  key={i}
                  className="inline-flex items-center text-xs px-2 py-1 rounded bg-white/10 hover:bg-white/20"
                  onClick={() => window.open(url, "_blank")}
                >
                  <ExternalLink className="w-3 h-3 mr-1" />
                  {previewUrls.length > 1 ? `查看第 ${i + 1} 页` : "查看原页"}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export type Reference = {
  id: number;
  text?: string;
  page?: number;
  citationId?: string;
  rank?: number;
  snippet?: string;
  images?: ImageItem[];
  tableImgPath?: string[];
};

export function MarkdownRenderer({
  content,
  references = [],
}: {
  content: string;
  references?: Reference[];
}) {
  const sanitizedContent = useMemo(
    () =>
      content
        .replace(/<img[\s\S]*?>/gi, "")
        .replace(/!\[[^\]]*]\(\s*(?:\.\/)?images\/[^)]+\)/gi, ""),
    [content]
  );

  return (
    <div className="space-y-3 text-foreground leading-relaxed prose prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
        components={{
          img: Img,
          code: Code,
          table: (p) => <table {...p} className="w-full border-collapse border border-border/30 rounded-lg overflow-hidden" />,
          thead: (p) => <thead {...p} className="bg-muted/30" />,
          th: (p) => <th {...p} className="px-3 py-2 border border-border/30 text-left font-medium" />,
          td: (p) => <td {...p} className="px-3 py-2 border border-border/30 text-sm" />,
          h1: (p) => <h1 {...p} className="text-2xl font-medium mt-4 mb-3" />,
          h2: (p) => <h2 {...p} className="text-xl font-medium mt-4 mb-2" />,
          h3: (p) => <h3 {...p} className="text-lg font-medium mt-3 mb-2" />,
          h4: (p) => <h4 {...p} className="text-base font-medium mt-3 mb-1.5" />,
          h5: (p) => <h5 {...p} className="text-sm font-medium mt-2 mb-1" />,
          h6: (p) => <h6 {...p} className="text-xs font-medium mt-2 mb-1" />,
          ul:  (p) => <ul {...p} className="list-disc pl-5 space-y-1" />,
          ol:  (p) => <ol {...p} className="list-decimal pl-5 space-y-1" />,
          a:   (p) => <a {...p} className="text-primary underline underline-offset-4" target="_blank" />,
        }}
      >
        {sanitizedContent}
      </ReactMarkdown>

      {/* 相关文档片段 */}
      {references?.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border/30">
          <div className="flex items-center gap-2 mb-2">
            <FileText className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">相关文档片段</span>
            <span className="text-xs text-muted-foreground">({references.length})</span>
          </div>
          <div className="space-y-2">
            {references
              .filter((r) => !!r.citationId)
              .map((r, i) => (
                <ReferenceCard
                  key={r.citationId!}
                  citationId={r.citationId!}
                  index={i}
                  fallbackImages={r.images}
                  fallbackTableImgPath={r.tableImgPath}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}