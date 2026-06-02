import { useState } from "react";
import { Header } from "./components/Header";
import { ChatInterface } from "./components/ChatInterface";
import { PDFPanel } from "./components/PDFPanel";
import { Toaster } from "./components/ui/sonner";

// ✨ 与 PDFPanel / ChatInterface 保持一致的类型
type ChatMode = 'pdf' | 'graph';

export default function App() {
  const [chatKey, setChatKey] = useState(0);
  const [currentFileId, setCurrentFileId] = useState<string>('');
  const [currentFileName, setCurrentFileName] = useState<string>('');
  const [currentTotalPages, setCurrentTotalPages] = useState<number>(0);

  // ✨ 新增:统一管理 PDF / 图谱 模式,作为单一数据源(single source of truth)
  // 由 PDFPanel 中的切换按钮触发更新,同时下发给 ChatInterface 决定问答模式
  const [chatMode, setChatMode] = useState<ChatMode>('pdf');

  // ✨ 新增:图谱可视化 Cypher,由图谱模式问答动态推送
  // 每次问答开始时,ChatInterface 收到 graph_query 事件 → 上抛到这里 → 下发给 PDFPanel
  const [graphCypher, setGraphCypher] = useState<string>("");

  const handleClearChat = () => {
    setChatKey((prev) => prev + 1);
  };

  const handleFileReady = (fileId: string, fileName: string, totalPages: number) => {
    setCurrentFileId(fileId);
    setCurrentFileName(fileName);
    setCurrentTotalPages(totalPages);
    // 新文件就绪时回到 PDF 模式,避免残留上一次的图谱模式
    setChatMode('pdf');
    // 切换文件时清空旧的图谱 Cypher
    setGraphCypher("");
  };

  // ✨ 新增:接收 PDFPanel 的模式切换事件
  const handleModeChange = (mode: ChatMode) => {
    setChatMode(mode);
  };

  // ✨ 新增:接收 ChatInterface 上抛的图谱 Cypher
  const handleGraphQueryReady = (cypher: string, nodeCount: number) => {
    console.log('🟢 App 收到 graph cypher,节点数:', nodeCount);
    setGraphCypher(cypher);
  };

  return (
    <div className="dark min-h-screen bg-background text-foreground relative overflow-hidden">
      {/* Enhanced background system */}
      <div className="background-system">
        {/* Floating orbs */}
        <div className="floating-elements">
          <div className="floating-orb floating-orb-1"></div>
          <div className="floating-orb floating-orb-2"></div>
          <div className="floating-orb floating-orb-3"></div>
          <div className="floating-orb floating-orb-4"></div>
        </div>

        {/* Geometric decorations */}
        <div className="geometric-decorations">
          <div className="geometric-line geometric-line-1"></div>
          <div className="geometric-line geometric-line-2"></div>
          <div className="geometric-line geometric-line-3"></div>
          <div className="geometric-polygon geometric-polygon-1"></div>
          <div className="geometric-polygon geometric-polygon-2"></div>
          <div className="geometric-circle geometric-circle-1"></div>
          <div className="geometric-circle geometric-circle-2"></div>
        </div>

        {/* Particle system */}
        <div className="particle-system">
          {Array.from({ length: 15 }).map((_, i) => (
            <div key={i} className={`particle particle-${i + 1}`}></div>
          ))}
        </div>

        {/* Light beams */}
        <div className="light-beams">
          <div className="light-beam light-beam-1"></div>
          <div className="light-beam light-beam-2"></div>
          <div className="light-beam light-beam-3"></div>
        </div>

        {/* Grid overlay */}
        <div className="grid-overlay"></div>
      </div>

      <div className="relative z-10">
        <div className="h-screen flex flex-col">
          {/* Header with reduced bottom margin */}
          <div className="max-w-7xl mx-auto w-full">
            <Header />
          </div>

          {/* Main Content - Consistent spacing layout */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-6 min-h-0 px-6 pb-6">
            {/* Left Column - Chat Interface (larger) */}
            <div className="flex flex-col min-h-0">
              <ChatInterface
                key={chatKey}
                onClearChat={handleClearChat}
                fileId={currentFileId}
                fileName={currentFileName}
                chatMode={chatMode}                              // ✨ 下发当前模式
                onGraphQueryReady={handleGraphQueryReady}        // ✨ 接收图谱 Cypher
              />
            </div>

            {/* Right Column - PDF Panel (smaller, aligned to right edge) */}
            <div className="flex flex-col min-h-0">
              <PDFPanel
                onFileReady={handleFileReady}
                viewMode={chatMode}             // ✨ 受控:模式来自父组件
                onModeChange={handleModeChange} // ✨ 切换时通知父组件
                graphCypher={graphCypher}       // ✨ 下发图谱 Cypher,触发 GraphView 切换
              />
            </div>
          </div>
        </div>
      </div>

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        expand={false}
        richColors
        closeButton
        theme="dark"
      />
    </div>
  );
}