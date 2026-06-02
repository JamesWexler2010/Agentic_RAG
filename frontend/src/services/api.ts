// API服务层 - 处理所有后端API调用
/// <reference types="vite/client" />
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001/api/v1';

export interface PdfUploadResponse {
  fileId: string;
  name: string;
  pages: number;
}

export interface ParseStatusResponse {
  status: 'idle' | 'parsing' | 'ready' | 'error';
  progress: number;
  errorMsg?: string;
}

export interface CitationChunk {
  id: string;
  fileId: string;
  page: number;
  snippet: string;
  bbox: [number, number, number, number];
  previewUrl: string;
  previewUrls?: string[]; // 新增。展示多页。可选的多页预览URL列表
  images?: Array<{ alt: string; url: string }>; // ✨ chunk 内提取的图片列表
}

export interface ChatReference {
  id: number;
  text: string;
  page: number;
  citationId?: string;
  rank?: number;
  snippet?: string;
}

// 健康检查
export async function checkHealth(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  } catch (error) {
    throw new Error('API unavailable');
  }
}

// PDF上传
export async function uploadPdf(file: File, replace = true): Promise<PdfUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('replace', replace.toString());

  const response = await fetch(`${API_BASE_URL}/pdf/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

// 开始PDF解析
export async function startParse(fileId: string): Promise<{ jobId: string }> {
  const response = await fetch(`${API_BASE_URL}/pdf/parse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId }),
  });

  if (!response.ok) throw new Error(`Parse start failed: ${response.statusText}`);
  return response.json();
}

// 查询解析状态
export async function getParseStatus(fileId: string): Promise<ParseStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/pdf/status?fileId=${encodeURIComponent(fileId)}`);
  if (!response.ok) throw new Error(`Status check failed: ${response.statusText}`);
  return response.json();
}

// 获取PDF页面图片
export function getPdfPageUrl(fileId: string, page: number, type: 'original' | 'parsed'): string {
  return `${API_BASE_URL}/pdf/page?fileId=${encodeURIComponent(fileId)}&page=${page}&type=${type}`;
}

// 获取Citation详情
export async function getCitationChunk(citationId: string): Promise<CitationChunk> {
  const response = await fetch(`${API_BASE_URL}/pdf/chunk?citationId=${encodeURIComponent(citationId)}`);
  if (!response.ok) throw new Error(`Citation fetch failed: ${response.statusText}`);
  return response.json();
}

// 构建向量索引
export async function buildIndex(fileId: string): Promise<{ ok: boolean; chunks: number }> {
  const response = await fetch(`${API_BASE_URL}/index/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId }),
  });
  if (!response.ok) throw new Error(`Index build failed: ${response.statusText}`);
  return response.json();
}

// 搜索索引
export async function searchIndex(fileId: string, query: string, k = 5): Promise<{
  ok: boolean;
  results: Array<{ text: string; score: number; metadata: Record<string, any> }>;
}> {
  const response = await fetch(`${API_BASE_URL}/index/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId, query, k }),
  });
  if (!response.ok) throw new Error(`Search failed: ${response.statusText}`);
  return response.json();
}

// ---------------------------------------------------------
// ✨ 以下是为历史文件列表功能新增的两个接口
// ---------------------------------------------------------

// 获取历史文件列表
export async function getHistoryFiles(): Promise<{ files: any[] }> {
  const response = await fetch(`${API_BASE_URL}/pdf/list`);
  if (!response.ok) {
    throw new Error(`Failed to fetch history files: ${response.statusText}`);
  }
  return response.json();
}

// 切换当前激活的文件
export async function selectFile(fileId: string): Promise<{ ok: boolean; current: any }> {
  const response = await fetch(`${API_BASE_URL}/pdf/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId }),
  });
  if (!response.ok) {
    throw new Error(`Failed to select file: ${response.statusText}`);
  }
  return response.json();
}

// 删除历史文件
export async function deleteFile(fileId: string): Promise<{ ok: boolean; deleted: string }> {
  const response = await fetch(`${API_BASE_URL}/pdf/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId }),
  });
  if (!response.ok) {
    throw new Error(`Failed to delete file: ${response.statusText}`);
  }
  return response.json();
}

// ---------------------------------------------------------
// 以下是原有 Chat Stream 相关接口保持不变
// ---------------------------------------------------------

// 自定义SSE处理函数
export async function processChatStream(
  message: string,
  onToken: (text: string) => void,
  onCitation: (citation: { citation_id: string; fileId: string; rank: number; page: number; previewUrl: string; snippet?: string; }) => void,
  onDone: (data: { used_retrieval: boolean }) => void,
  onError: (error: string) => void,
  pdfFileId?: string,
  sessionId = 'default',
  mode: 'pdf' | 'graph' = 'pdf',
  onGraphQuery?: (data: { cypher: string; node_count: number }) => void  // ✨ 新增
){
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        message,
        sessionId,
        ...(pdfFileId && { pdfFileId }),
        mode, // ✨ 2. 将 mode 传递给后端
      }),
    });

    // ... 后面的 reader.read() 逻辑保持完全不变 ...
    if (!response.ok) throw new Error(`Chat request failed: ${response.statusText}`);
    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const event of events) {
        if (!event.trim()) continue;
        const lines = event.split('\n');
        let eventType = '';
        let eventData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.substring(7);
          else if (line.startsWith('data: ')) eventData = line.substring(6);
        }

        if (eventType && eventData) {
          try {
            const data = JSON.parse(eventData);
            switch (eventType) {
              case 'citation': onCitation(data); break;
              case 'token': onToken(data.text); break;
              case 'done': onDone(data); return;
              case 'error': onError(data.message || 'Unknown error'); return;
              case 'graph_query':    // ✨ 新增
                onGraphQuery?.(data);
                break;
            }
          } catch (e) { console.error('Failed to parse SSE data:', e); }
        }
      }
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      const mockResponse = `I understand you're asking about: "${message}".\n\nSince the backend API is not currently available, I'm showing you a demonstration of the interface.`;
      const words = mockResponse.split(' ');
      let currentIndex = 0;
      
      const streamInterval = setInterval(() => {
        if (currentIndex < words.length) {
          onToken(words[currentIndex] + ' ');
          currentIndex++;
        } else {
          clearInterval(streamInterval);
          onDone({ used_retrieval: false });
        }
      }, 50);
      return;
    }
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

export async function clearSession(sessionId = 'default'): Promise<{ ok: boolean; sessionId: string; cleared: boolean; }> {
  const response = await fetch(`${API_BASE_URL}/chat/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId }),
  });
  if (!response.ok) throw new Error(`Clear session failed: ${response.statusText}`);
  return response.json();
}