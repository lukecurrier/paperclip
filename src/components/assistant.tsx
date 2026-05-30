"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  UploadCloud,
  BookOpen,
  MessageSquare,
  FileText,
  BookOpenCheck,
} from "lucide-react";
import SimpleModelSelector from "./simplemodelselector";
import { useRef } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

const LoadingBar = ({
  progress = 0,
  message = "",
}: {
  progress: number;
  message: string;
}) => (
  <div className="mt-4 p-4 bg-gray-50 rounded-lg">
    <div className="flex justify-between items-center mb-2">
      <span className="text-sm text-gray-600">{message}</span>
      <span className="text-sm text-gray-600">
        {Math.round(progress * 100)}%
      </span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2.5">
      <div
        className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
        style={{ width: `${progress * 100}%` }}
      ></div>
    </div>
  </div>
);

const Assistant = () => {
  const [file, setFile] = useState<File | null>(null);
  const [paperId, setPaperId] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMessage, setProcessingMessage] = useState("");
  const [currentTab, setCurrentTab] = useState("upload");
  const [paperContent, setPaperContent] = useState("");
  const [summary, setSummary] = useState("");
  const [messages, setMessages] = useState<
    Array<{ sender: string; content: string }>
  >([]);
  const [userMessage, setUserMessage] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [isRegeneratingSummary, setIsRegeneratingSummary] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gpt-4o-mini");
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const [isThinking, setIsThinking] = useState(false);

  useEffect(() => {
    const savedModel = localStorage.getItem("selectedModel");
    if (savedModel) {
      setSelectedModel(savedModel);
    }
  }, []);

  // Update localStorage when model changes
  useEffect(() => {
    localStorage.setItem("selectedModel", selectedModel);
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [selectedModel, messages]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (uploadedFile) {
      if (!uploadedFile.type.includes("pdf")) {
        alert("Please upload a PDF file");
        return;
      }

      setFile(uploadedFile);

      const paperId = uploadedFile.name.replace(/\.[^/.]+$/, "");
      setPaperId(paperId);
    }
  };

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const regenerateSummary = async () => {
    if (!paperId) return;

    setIsRegeneratingSummary(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/regenerate-summary`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          paperId,
          modelId: selectedModel,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to regenerate summary: ${response.status}`);
      }

      const data = await response.json();

      // update UI with new summary
      setSummary(data.summary);
    } catch (err) {
      console.error(err);
    } finally {
      setIsRegeneratingSummary(false);
    }
  };

  const processPaper = async () => {
    if (!file || !paperId) return;

    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingMessage("Uploading PDF...");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("paperId", paperId);
      formData.append("modelId", selectedModel);

      const response = await fetch(`${API_BASE_URL}/api/process-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Unknown error");
      }

      // ---------------------------
      // REAL BACKEND POLLING ONLY
      // ---------------------------

      let pollInterval: NodeJS.Timeout | null = null;
      let isDone = false;

      const pollStatus = async () => {
        try {
          const res = await fetch(
            `${API_BASE_URL}/api/check-progress/${paperId}`,
          );

          if (!res.ok) return;

          const status = await res.json();

          setProcessingProgress(status.progress ?? 0);
          setProcessingMessage(status.message ?? "Processing...");

          if (status.status === "complete") {
            isDone = true;

            if (pollInterval) clearInterval(pollInterval);

            setProcessingProgress(1);
            setProcessingMessage("Loading paper...");

            const paperRes = await fetch(
              `${API_BASE_URL}/api/paper/${paperId}`,
            );
            const paperData = await paperRes.json();

            setPaperContent(paperData.content || "");
            setSummary(paperData.summary || "");
            setPdfUrl(`${API_BASE_URL}/api/pdf/${paperId}?t=${Date.now()}`);

            setTimeout(() => {
              setCurrentTab("summary");
              setIsProcessing(false);
            }, 500);
          }

          if (status.status === "failed") {
            if (pollInterval) clearInterval(pollInterval);

            setIsProcessing(false);
            setProcessingMessage(status.message || "Failed");
            alert("Processing failed. Please try again.");
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      };

      // start polling immediately
      pollInterval = setInterval(pollStatus, 2000);

      // optional safety timeout
      setTimeout(() => {
        if (!isDone && pollInterval) {
          clearInterval(pollInterval);
          setIsProcessing(false);
          alert("Processing is taking too long. Please try again later.");
        }
      }, 180000);
    } catch (error) {
      console.error("Error processing PDF:", error);
      alert("Failed to process PDF. Please try again.");
      setIsProcessing(false);
      setProcessingProgress(0);
      setProcessingMessage("");
    }
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim()) return;

    const newMessages = [...messages, { sender: "user", content: userMessage }];

    setMessages(newMessages);
    const currentMessage = userMessage;
    setUserMessage("");

    // ✅ show loader
    setIsThinking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: currentMessage,
          paperId: paperId,
          modelId: selectedModel,
        }),
      });

      const data = await response.json();

      setMessages([
        ...newMessages,
        {
          sender: "ai",
          content: data.response,
        },
      ]);
    } catch (error) {
      console.error("Error getting response:", error);

      setMessages([
        ...newMessages,
        {
          sender: "ai",
          content:
            "I'm sorry, I encountered an error processing your question.",
        },
      ]);
    } finally {
      // ✅ hide loader
      setIsThinking(false);
    }
  };

  // Handle keyboard events in the textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If Enter is pressed without holding Cmd (or Ctrl), send the message
    if (e.key === "Enter" && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
      e.preventDefault(); // Prevent default behavior (new line)
      handleSendMessage();
    }
    // If Cmd+Enter is pressed, allow new line
    // No need to do anything as the default behavior will add a new line
  };

  return (
    <div className="flex flex-col w-full max-w-6xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-2xl">PaperClip</CardTitle>
            <CardDescription>
              Upload an AI research paper to summarize and discuss
            </CardDescription>
          </div>
          <SimpleModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
          />
        </CardHeader>

        <CardContent>
          <Tabs value={currentTab} onValueChange={setCurrentTab}>
            <TabsList className="grid grid-cols-3 mb-6">
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="summary" disabled={!summary}>
                Summary
              </TabsTrigger>
              <TabsTrigger value="discuss" disabled={!summary}>
                Discuss
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-4">
              {isProcessing && (
                <LoadingBar
                  progress={processingProgress}
                  message={processingMessage}
                />
              )}

              <div className="border-2 border-dashed rounded-lg p-12 text-center">
                <div className="flex flex-col items-center gap-2">
                  <UploadCloud className="h-10 w-10 text-gray-400" />
                  <p className="text-sm text-gray-500">
                    Drag and drop or click to upload a PDF
                  </p>
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="application/pdf"
                    onChange={handleFileUpload}
                  />
                  <Button
                    variant="outline"
                    onClick={() =>
                      document.getElementById("file-upload")?.click()
                    }
                    className="mt-2"
                  >
                    Select File
                  </Button>
                </div>
              </div>

              {file && (
                <div className="bg-gray-100 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-blue-500" />
                      <span>{file.name}</span>
                    </div>
                    <Button
                      onClick={processPaper}
                      disabled={isProcessing}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {isProcessing ? "Processing..." : "Process Paper"}
                    </Button>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="summary" className="space-y-4">
              <div className="space-y-6">
                <div className="border rounded-lg p-6 shadow-sm bg-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <BookOpen className="h-6 w-6 text-blue-600" />
                      <h2 className="text-xl font-bold">Paper Summary</h2>
                    </div>
                  </div>

                  {!summary ? (
                    <div className="flex items-center justify-center h-40 bg-gray-50 rounded-md">
                      <p className="text-gray-500">Loading summary...</p>
                    </div>
                  ) : (
                    <div className="prose prose-blue max-w-none">
                      {summary.split("\n").map((line, i) => {
                        // Handle bold text with asterisks (like **text**)
                        const processedLine = line.replace(
                          /\*\*(.*?)\*\*/g,
                          "<strong>$1</strong>",
                        );

                        // Handle headings
                        if (line.startsWith("# ")) {
                          return (
                            <h1
                              key={i}
                              className="text-2xl font-bold mt-6 mb-4 text-blue-800"
                            >
                              {line.substring(2)}
                            </h1>
                          );
                        } else if (line.startsWith("## ")) {
                          return (
                            <h2
                              key={i}
                              className="text-xl font-bold mt-5 mb-3 text-blue-700"
                            >
                              {line.substring(3)}
                            </h2>
                          );
                        } else if (line.startsWith("### ")) {
                          return (
                            <h3
                              key={i}
                              className="text-lg font-bold mt-4 mb-2 text-blue-600"
                            >
                              {line.substring(4)}
                            </h3>
                          );
                        }
                        // Handle numbered lists (looking for patterns like "1. ", "2. ", etc.)
                        else if (/^\d+\.\s/.test(line)) {
                          // Extract the number and the content
                          const match = line.match(/^(\d+)\.\s(.*)$/);
                          if (match) {
                            const [_, number, content] = match;
                            // Use dangerouslySetInnerHTML to parse any bold formatting within the list item
                            return (
                              <div key={i} className="flex gap-2 my-1">
                                <span className="font-semibold">{number}.</span>
                                <span
                                  dangerouslySetInnerHTML={{
                                    __html: content.replace(
                                      /\*\*(.*?)\*\*/g,
                                      "<strong>$1</strong>",
                                    ),
                                  }}
                                />
                              </div>
                            );
                          }
                          return (
                            <p
                              key={i}
                              className="my-2 text-gray-700 leading-relaxed"
                            >
                              {line}
                            </p>
                          );
                        }
                        // Handle bullet points
                        else if (
                          line.startsWith("* ") ||
                          line.startsWith("- ")
                        ) {
                          return (
                            <div key={i} className="flex gap-2 my-1 ml-5">
                              <span>•</span>
                              <span
                                dangerouslySetInnerHTML={{
                                  __html: line
                                    .substring(2)
                                    .replace(
                                      /\*\*(.*?)\*\*/g,
                                      "<strong>$1</strong>",
                                    ),
                                }}
                              />
                            </div>
                          );
                        } else if (
                          line.startsWith("  * ") ||
                          line.startsWith("  - ")
                        ) {
                          return (
                            <div key={i} className="flex gap-2 my-1 ml-10">
                              <span>•</span>
                              <span
                                dangerouslySetInnerHTML={{
                                  __html: line
                                    .substring(4)
                                    .replace(
                                      /\*\*(.*?)\*\*/g,
                                      "<strong>$1</strong>",
                                    ),
                                }}
                              />
                            </div>
                          );
                        }
                        // Handle code blocks
                        else if (line.startsWith("```")) {
                          return (
                            <div
                              key={i}
                              className="bg-gray-100 p-2 rounded my-2 font-mono text-sm"
                            >
                              {line.substring(3)}
                            </div>
                          );
                        } else if (line.startsWith("`") && line.endsWith("`")) {
                          return (
                            <code
                              key={i}
                              className="bg-gray-100 px-1 rounded text-sm font-mono"
                            >
                              {line.substring(1, line.length - 1)}
                            </code>
                          );
                        }
                        // Handle blockquotes
                        else if (line.startsWith("> ")) {
                          return (
                            <blockquote
                              key={i}
                              className="border-l-4 border-gray-300 pl-4 italic my-2"
                              dangerouslySetInnerHTML={{
                                __html: line
                                  .substring(2)
                                  .replace(
                                    /\*\*(.*?)\*\*/g,
                                    "<strong>$1</strong>",
                                  ),
                              }}
                            ></blockquote>
                          );
                        }
                        // Handle empty lines
                        else if (line.trim() === "") {
                          return <div key={i} className="my-2"></div>;
                        }
                        // Handle regular paragraphs
                        else {
                          return (
                            <p
                              key={i}
                              className="my-2 text-gray-700 leading-relaxed"
                              dangerouslySetInnerHTML={{
                                __html: line.replace(
                                  /\*\*(.*?)\*\*/g,
                                  "<strong>$1</strong>",
                                ),
                              }}
                            ></p>
                          );
                        }
                      })}
                    </div>
                  )}
                </div>

                <div className="flex justify-between">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentTab("upload")}
                    className="flex items-center gap-2"
                  >
                    <UploadCloud className="h-4 w-4" />
                    Upload Different Paper
                  </Button>

                  <Button
                    onClick={() => setCurrentTab("discuss")}
                    className="bg-blue-600 hover:bg-blue-700 flex items-center gap-2"
                  >
                    <MessageSquare className="h-4 w-4" />
                    Discuss This Paper
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="discuss" className="space-y-4">
              <div className="flex gap-4 h-[600px]">
                {/* PDF Viewer */}
                <div className="w-3/5 border rounded-lg overflow-hidden">
                  {pdfUrl ? (
                    <iframe
                      src={`${pdfUrl}#toolbar=0&view=FitH`}
                      className="w-full h-full"
                      title="Paper PDF"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full bg-gray-100">
                      <p className="text-gray-500">PDF not available</p>
                    </div>
                  )}
                </div>

                {/* Chat Interface */}
                <div className="w-3/5 flex flex-col">
                  <div className="border rounded-lg p-4 flex-grow overflow-y-auto flex flex-col space-y-4">
                    {messages.length === 0 ? (
                      <div className="text-center text-gray-500 my-auto">
                        <MessageSquare className="h-12 w-12 mx-auto opacity-30" />
                        <p className="mt-2">Ask questions about the paper</p>
                        <p className="text-sm mt-1">
                          Press Enter to send, Cmd+Enter for a new line
                        </p>
                      </div>
                    ) : (
                      messages.map((msg, i) => (
                        <div
                          ref={chatEndRef}
                          key={i}
                          className={`${msg.sender === "user" ? "ml-auto" : "mr-auto"} max-w-[85%]`}
                        >
                          <div
                            className={`p-3 rounded-lg ${
                              msg.sender === "user"
                                ? "bg-blue-600 text-white"
                                : "bg-gray-100 text-gray-800"
                            }`}
                          >
                            {msg.content}
                          </div>
                        </div>
                      ))
                    )}
                    {isThinking && (
                      <div className="flex items-center gap-2 text-gray-500 animate-pulse">
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:150ms]" />
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:300ms]" />
                        <span className="text-sm">
                          PaperClip is thinking...
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="flex gap-2 mt-4">
                    <Textarea
                      placeholder="Ask about the paper... (Press Enter to send, Cmd+Enter for a new line)"
                      value={userMessage}
                      onChange={(e) => setUserMessage(e.target.value)}
                      onKeyDown={handleKeyDown}
                      className="flex-1 min-h-16"
                    />
                    <Button
                      onClick={handleSendMessage}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      Send
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default Assistant;
