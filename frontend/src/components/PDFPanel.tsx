import { useState, useRef, useEffect } from "react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Progress } from "./ui/progress";
import { ScrollArea } from "./ui/scroll-area";
import {
  Upload,
  FileText,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  File,
  Library,
  Trash2,
  Network,          // ✨ 图谱图标
  FileSearch        // ✨ PDF 图标(切回时用)
} from "lucide-react";
import {
  uploadPdf, startParse, getParseStatus, buildIndex, getPdfPageUrl,
  getHistoryFiles, selectFile,
  deleteFile
} from "../services/api";
import { toast } from "sonner";
import { GraphView } from "./Neo4j";   // 引入图谱组件

type UploadStatus = 'idle' | 'uploading' | 'parsing' | 'ready' | 'error';
type ViewMode = 'pdf' | 'graph';   // 视图 / 问答模式

interface HistoryFile {
  fileId: string;
  name: string;
  pages: number;
  hasIndex: boolean;
}

interface PDFPanelProps {
  className?: string;
  onFileReady?: (fileId: string, fileName: string, totalPages: number) => void;
  // ✨ 受控模式相关 props——由 App.tsx 统一管理 viewMode
  // 兼容性:两个 prop 都设为可选,未传入时回退到旧的内部 state 行为
  viewMode?: ViewMode;
  onModeChange?: (mode: ViewMode) => void;
  // ✨ 新增:由 App.tsx 下发的图谱可视化 Cypher
  //    每次图谱模式问答完成后,Cypher 会通过这个 prop 传进来,
  //    PDFPanel 直接透传给 GraphView,触发自动切换图谱视图。
  graphCypher?: string;
}

export function PDFPanel({
  className,
  onFileReady,
  viewMode: viewModeProp,
  onModeChange,
  graphCypher,   // ✨ 新增:从 App 接收图谱 Cypher,透传给 GraphView
}: PDFPanelProps) {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [fileName, setFileName] = useState<string>('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileId, setFileId] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const [historyFiles, setHistoryFiles] = useState<HistoryFile[]>([]);
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);

  // ✨ 内部 fallback state:仅在父组件未传入 viewMode 时使用(保持向后兼容)
  const [internalViewMode, setInternalViewMode] = useState<ViewMode>('pdf');

  // ✨ 受控/非受控统一出口:有 prop 用 prop,否则用内部 state
  const isControlled = viewModeProp !== undefined;
  const viewMode: ViewMode = isControlled ? (viewModeProp as ViewMode) : internalViewMode;

  // ✨ 设置 viewMode 的统一入口:受控模式下走 onModeChange,否则改内部 state
  const setViewMode = (next: ViewMode) => {
    if (isControlled) {
      onModeChange?.(next);
    } else {
      setInternalViewMode(next);
    }
  };

  const fileInputRef = useRef<HTMLInputElement>(null);
  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  // 加载历史列表
  const loadHistory = async () => {
    try {
      const data = await getHistoryFiles();
      setHistoryFiles(data.files || []);
      setIsLibraryOpen(true);
    } catch (error) {
      toast.error("Failed to load file library");
    }
  };

  // 选择并激活历史文件
  const handleSelectHistory = async (file: HistoryFile) => {
    try {
      const res = await selectFile(file.fileId);
      if (res.ok) {
        setFileId(file.fileId);
        setFileName(file.name);
        setTotalPages(file.pages);
        setCurrentPage(1);
        setUploadStatus('ready');
        setIsLibraryOpen(false);
        setViewMode('pdf');   // ✨ 切换文件时回到 PDF 视图(同步通知父组件)
        toast.success(`Loaded: ${file.name}`);
        onFileReady?.(file.fileId, file.name, file.pages);
      }
    } catch (error) {
      toast.error("Failed to switch file");
    }
  };

  // 删除历史文件
  const handleDeleteHistory = async (e: React.MouseEvent, file: HistoryFile) => {
    e.stopPropagation();
    if (!confirm(`确认删除「${file.name}」？此操作不可撤销。`)) return;

    try {
      await deleteFile(file.fileId);
      if (fileId === file.fileId) {
        handleReplace();
      }
      setHistoryFiles(prev => prev.filter(f => f.fileId !== file.fileId));
      toast.success(`已删除: ${file.name}`);
    } catch (error) {
      toast.error("删除失败");
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleReplace = () => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
    }
    setUploadStatus('idle');
    setFileName('');
    setCurrentPage(1);
    setTotalPages(0);
    setUploadProgress(0);
    setFileId('');
    setErrorMessage('');
    setIsLibraryOpen(false);
    setViewMode('pdf');   // ✨ 重置视图(同步通知父组件)
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ✨ 切换视图模式:走统一的 setViewMode,自动同步给 App.tsx
  const toggleViewMode = () => {
    setViewMode(viewMode === 'pdf' ? 'graph' : 'pdf');
  };

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
      case 'parsing':
        return <Loader2 className="w-4 h-4 animate-spin" />;
      case 'ready':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getStatusText = () => {
    switch (uploadStatus) {
      case 'uploading': return 'Uploading...';
      case 'parsing': return 'Parsing...';
      case 'ready': return 'Ready';
      case 'error': return 'Error';
      default: return 'No document';
    }
  };

  const getStatusVariant = (): "default" | "secondary" | "destructive" | "outline" => {
    switch (uploadStatus) {
      case 'ready': return 'default';
      case 'error': return 'destructive';
      case 'uploading':
      case 'parsing': return 'secondary';
      default: return 'outline';
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !file.type.includes('pdf')) {
      toast.error('Please select a valid PDF file');
      return;
    }
    setFileName(file.name);
    setUploadStatus('uploading');
    setUploadProgress(0);
    setErrorMessage('');
    try {
      const uploadResponse = await uploadPdf(file);
      setFileId(uploadResponse.fileId);
      setTotalPages(uploadResponse.pages);
      setCurrentPage(1);
      toast.success('PDF uploaded successfully');
      setUploadStatus('parsing');
      await startParse(uploadResponse.fileId);
      startStatusPolling(uploadResponse.fileId);
    } catch (error) {
      setUploadStatus('error');
      setErrorMessage('Upload failed');
    }
  };

  const startStatusPolling = (fileId: string) => {
    statusCheckInterval.current = setInterval(async () => {
      try {
        const status = await getParseStatus(fileId);
        setUploadProgress(status.progress);
        if (status.status === 'ready') {
          setUploadStatus('ready');
          clearInterval(statusCheckInterval.current!);
          await buildIndex(fileId);
          onFileReady?.(fileId, fileName, totalPages);
        }
      } catch (e) { clearInterval(statusCheckInterval.current!); }
    }, 2000);
  };

  const nextPage = () => currentPage < totalPages && setCurrentPage(prev => prev + 1);
  const prevPage = () => currentPage > 1 && setCurrentPage(prev => prev - 1);

  useEffect(() => () => { if (statusCheckInterval.current) clearInterval(statusCheckInterval.current); }, []);

  return (
    <div className={`glass-panel-bright h-full flex flex-col relative overflow-hidden ${className}`}>
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-gradient-to-br from-green-500/20 via-transparent to-blue-500/20"></div>
      </div>

      <div className="relative p-5 border-b border-border/80">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-green-500/15 border border-green-500/30 shadow-lg">
              <File className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <h2 className="elegant-title text-base">Document</h2>
              <p className="text-xs text-muted-foreground/80 mt-1">
                {viewMode === 'graph' ? 'Knowledge Graph' : 'PDF Analysis'}
              </p>
            </div>
          </div>

          {/* 右侧:切换按钮 + Ready 标识 */}
          <div className="flex items-center gap-2">
            {/* 切换按钮:只在 ready 状态下显示 */}
            {uploadStatus === 'ready' && (
              <Button
                variant="outline"
                size="sm"
                onClick={toggleViewMode}
                className={`h-8 px-3 gap-1.5 transition-all ${
                  viewMode === 'graph'
                    ? 'bg-blue-500/15 border-blue-500/40 text-blue-500 hover:bg-blue-500/25'
                    : 'bg-secondary/40 border-border/60 hover:bg-secondary/60'
                }`}
                title={viewMode === 'pdf' ? '切换到图谱视图(问答将走图谱 Agent)' : '切换到 PDF 视图(问答将走向量检索)'}
              >
                {viewMode === 'pdf' ? (
                  <>
                    <Network className="w-3.5 h-3.5" />
                    <span className="text-xs">图谱</span>
                  </>
                ) : (
                  <>
                    <FileSearch className="w-3.5 h-3.5" />
                    <span className="text-xs">PDF</span>
                  </>
                )}
              </Button>
            )}

            <Badge variant={getStatusVariant()} className="flex items-center gap-2 px-3 py-1 shadow-sm">
              {getStatusIcon()}
              <span className="text-xs">{getStatusText()}</span>
            </Badge>
          </div>
        </div>

        {uploadStatus === 'idle' ? (
          <div className="flex gap-2">
            <Button
              onClick={handleUploadClick}
              className="flex-1 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white shadow-lg border border-green-500/30 rounded-xl transition-all duration-200 min-h-[48px] h-[48px] text-base font-medium"
            >
              <Upload className="w-5 h-5 mr-2 flex-shrink-0" />
              <span className="flex-shrink-0">Upload PDF</span>
            </Button>

            <Button
              variant="outline"
              onClick={loadHistory}
              className="px-4 border-border/60 bg-secondary/20 hover:bg-secondary/40 rounded-xl min-h-[48px] h-[48px]"
              title="Open Library"
            >
              <Library className="w-5 h-5 text-primary" />
              <span>历史库</span>
            </Button>
          </div>
        ) : (
          <div className="flex gap-2">
            <div className="flex-1 text-sm text-muted-foreground truncate bg-secondary/40 p-3 rounded-lg border border-border/40 min-h-[48px] flex items-center">
              {fileName}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReplace}
              className="shrink-0 min-h-[48px] h-[48px] w-[48px] p-0 border-border/40 hover:bg-destructive/10"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        )}

        {(uploadStatus === 'uploading' || uploadStatus === 'parsing') && (
          <div className="mt-4">
            <Progress value={uploadProgress} className="h-2" />
            <p className="text-xs text-muted-foreground/80 mt-2">
              {uploadStatus === 'uploading' ? `Uploading... ${uploadProgress}%` : `Processing... ${uploadProgress}%`}
            </p>
          </div>
        )}
      </div>

      {uploadStatus === 'ready' ? (
        <div className="flex-1 flex flex-col relative min-h-0">
          {/* PDF 视图:原有的 Original / Parsed Tabs */}
          {viewMode === 'pdf' ? (
            <>
              <Tabs defaultValue="original" className="flex-1 flex flex-col min-h-0">
                <div className="px-5 pt-4">
                  <TabsList className="grid w-full grid-cols-2 h-10 bg-secondary/40 border border-border/40">
                    <TabsTrigger value="original" className="text-xs px-2 py-2 data-[state=active]:bg-primary/15 data-[state=active]:text-primary transition-all">Original</TabsTrigger>
                    <TabsTrigger value="parsed" className="text-xs px-2 py-2 data-[state=active]:bg-primary/15 data-[state=active]:text-primary transition-all">Parsed</TabsTrigger>
                  </TabsList>
                </div>
                <TabsContent value="original" className="flex-1 flex flex-col mt-4 mx-5 mb-4 min-h-0">
                  <div className="flex-1 bg-slate-900/40 border border-border/60 rounded-xl flex items-center justify-center shadow-inner overflow-hidden relative">
                    {fileId && <img key={`${fileId}-${currentPage}`} src={getPdfPageUrl(fileId, currentPage, 'original')} alt="PDF" className="max-w-full max-h-full object-contain" />}
                  </div>
                </TabsContent>
                <TabsContent value="parsed" className="flex-1 flex flex-col mt-4 mx-5 mb-4 min-h-0">
                  <div className="flex-1 bg-slate-900/40 border border-border/60 rounded-xl flex items-center justify-center shadow-inner overflow-hidden relative">
                    {fileId && (
                      <img
                        key={`${fileId}-${currentPage}-parsed`}
                        src={getPdfPageUrl(fileId, currentPage, 'parsed')}
                        alt="Parsed PDF"
                        className="max-w-full max-h-full object-contain"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none';
                        }}
                        onLoad={(e) => {
                          (e.target as HTMLImageElement).style.display = 'block';
                        }}
                      />
                    )}
                  </div>
                </TabsContent>
              </Tabs>

              {/* PDF 视图的页码导航 */}
              <div className="p-5 border-t border-border/60 bg-card/40">
                <div className="flex items-center justify-between">
                  <Button variant="outline" size="sm" onClick={prevPage} disabled={currentPage <= 1} className="h-10 px-4"><ChevronLeft className="w-4 h-4" /></Button>
                  <span className="text-sm text-muted-foreground font-medium">Page {currentPage} of {totalPages}</span>
                  <Button variant="outline" size="sm" onClick={nextPage} disabled={currentPage >= totalPages} className="h-10 px-4"><ChevronRight className="w-4 h-4" /></Button>
                </div>
              </div>
            </>
          ) : (
            /* 图谱视图 */
            <>
              <div className="flex-1 mt-4 mx-5 mb-4 min-h-0 bg-slate-900/40 border border-border/60 rounded-xl overflow-hidden shadow-inner">
                <GraphView fileId={fileId} externalCypher={graphCypher} />
              </div>
              <div className="p-5 border-t border-border/60 bg-card/40">
                <div className="text-xs text-muted-foreground/70 text-center">
                  拖拽节点调整布局 · 滚轮缩放 · 悬停查看详情
                </div>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center p-8 relative">
          {isLibraryOpen && historyFiles.length > 0 ? (
            <div className="w-full h-full flex flex-col animate-in fade-in zoom-in duration-300">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold">Library</h3>
                <Button variant="ghost" size="sm" onClick={() => setIsLibraryOpen(false)}>Back</Button>
              </div>
              <ScrollArea className="flex-1 rounded-xl border border-border/40 bg-secondary/10 p-2">
                <div className="space-y-2">
                  {historyFiles.map((file) => (
                    <div
                      key={file.fileId}
                      onClick={() => handleSelectHistory(file)}
                      className="group flex items-center justify-between p-3 rounded-lg hover:bg-primary/10 border border-transparent cursor-pointer transition-all"
                    >
                      <div className="flex items-center gap-3 overflow-hidden">
                        <FileText className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm truncate font-medium">{file.name}</span>
                        <span className="text-xs text-muted-foreground/60 truncate">{file.fileId}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-[10px]">{file.pages}P</Badge>
                        <button
                          onClick={(e) => handleDeleteHistory(e, file)}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/20 transition-all"
                          title="删除"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-destructive" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          ) : (
            <div className="text-center space-y-6 max-w-sm">
              <div className="w-20 h-20 bg-gradient-to-br from-green-500/15 to-blue-500/15 rounded-full flex items-center justify-center mx-auto border border-green-500/30 shadow-lg">
                <Upload className="w-10 h-10 text-green-500/80" />
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-foreground">No document uploaded</h3>
                <p className="text-sm text-muted-foreground/80">Upload a PDF or select from library.</p>
              </div>
            </div>
          )}
        </div>
      )}
      <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
    </div>
  );
}