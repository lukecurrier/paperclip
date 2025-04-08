"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { UploadCloud, BookOpen, MessageSquare, FileText, ArrowRight, BookOpenCheck } from 'lucide-react';

// API base URL - change this if your Flask server runs on a different port
const API_BASE_URL = 'http://127.0.0.1:5000';

const LoadingBar = ({ progress = 0, message = '' }: { progress: number; message: string }) => (
  <div className="mt-4 p-4 bg-gray-50 rounded-lg">
    <div className="flex justify-between items-center mb-2">
      <span className="text-sm text-gray-600">{message}</span>
      <span className="text-sm text-gray-600">{Math.round(progress * 100)}%</span>
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
  const [paperId, setPaperId] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMessage, setProcessingMessage] = useState('');
  const [currentTab, setCurrentTab] = useState('upload');
  const [paperContent, setPaperContent] = useState('');
  const [summary, setSummary] = useState('');
  const [messages, setMessages] = useState<Array<{sender: string, content: string}>>([]);
  const [userMessage, setUserMessage] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [isRegeneratingSummary, setIsRegeneratingSummary] = useState(false);

  useEffect(() => {
    // Check for paper data if we already have a paperId
    const checkForExistingPaper = async () => {
      if (paperId) {
        const exists = await checkExistingPaper(paperId);
        if (exists) {
          // Fetch the paper data
          try {
            const response = await fetch(`${API_BASE_URL}/api/paper/${paperId}`);
            if (response.ok) {
              const data = await response.json();
              setPaperContent(data.content || '');
              setSummary(data.summary || '');
              setPdfUrl(`${API_BASE_URL}/api/pdf/${paperId}`);
              setCurrentTab('discuss');
            }
          } catch (error) {
            console.error('Error fetching paper data:', error);
          }
        }
      }
    };
    
    checkForExistingPaper();
  }, [paperId]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (uploadedFile) {
      // Check if the file is a PDF
      if (!uploadedFile.type.includes('pdf')) {
        alert('Please upload a PDF file');
        return;
      }
      
      // Set the file
      setFile(uploadedFile);
      
      // Generate a paper ID from the filename (remove extension)
      const paperId = uploadedFile.name.replace(/\.[^/.]+$/, "");
      setPaperId(paperId);
    }
  };

  const checkExistingPaper = async (paperId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/check-paper/${paperId}`);
      if (response.ok) {
        const data = await response.json();
        return data.exists;
      }
      return false;
    } catch (error) {
      console.error('Error checking existing paper:', error);
      return false;
    }
  };

  const regenerateSummary = async () => {
    if (!paperId) return;
    
    setIsRegeneratingSummary(true);
    
    try {
      // Call the API to regenerate the summary
      const response = await fetch(`${API_BASE_URL}/api/regenerate-summary/${paperId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to regenerate summary: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setSummary(data.summary);
      } else {
        alert(`Error: ${data.error || 'Failed to regenerate summary'}`);
      }
    } catch (error) {
      console.error('Error regenerating summary:', error);
      alert('Failed to regenerate summary. Please try again.');
    } finally {
      setIsRegeneratingSummary(false);
    }
  };

  const processPaper = async () => {
    if (!file || !paperId) return;
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingMessage('Checking if paper already exists...');
    
    try {
      // Check if paper already exists
      const paperExists = await checkExistingPaper(paperId);
      
      if (paperExists) {
        setProcessingProgress(0.2);
        setProcessingMessage('Paper already processed, loading content...');
        
        // Fetch existing paper data
        const response = await fetch(`${API_BASE_URL}/api/paper/${paperId}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        setProcessingProgress(0.6);
        setProcessingMessage('Loading paper content...');
        
        const data = await response.json();
        setPaperContent(data.content);
        setSummary(data.summary);
        setPdfUrl(`${API_BASE_URL}/api/pdf/${paperId}`);
        
        setProcessingProgress(0.9);
        setProcessingMessage('Preparing interface...');
        
        setTimeout(() => {
          setProcessingProgress(1);
          setProcessingMessage('Paper loaded successfully!');
          setCurrentTab('discuss'); // Go directly to the discuss tab
        }, 500);
      } else {
        // Paper doesn't exist, process it
        // Create form data for file upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('paperId', paperId);
        
        // Call the backend API to process the PDF
        const response = await fetch(`${API_BASE_URL}/api/process-pdf`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
          throw new Error(data.error || 'Unknown error processing PDF');
        }
        
        // Simulated loading stages based on the logs, but we'll check real progress too
        const loadingStages = [
          { progress: 0.05, message: 'Uploading PDF...', time: 1000 },
          { progress: 0.1, message: 'Loading layout model...', time: 2000 },
          { progress: 0.15, message: 'Loading recognition models...', time: 3000 },
          { progress: 0.2, message: 'Starting PDF conversion...', time: 1000 },
          { progress: 0.3, message: 'Recognizing layout...', time: 4000 },
          { progress: 0.4, message: 'Running OCR error detection...', time: 2000 },
          { progress: 0.5, message: 'Detecting text boundaries...', time: 2000 },
          { progress: 0.6, message: 'Recognizing text (this may take a while)...', time: 20000 }
          // We'll skip the rest and check real progress
        ];
        
        // Start checking real progress after simulated stages complete
        let checkProgressTimer: NodeJS.Timeout | null = null;
        let finalStageReached = false;
        
        const checkRealProgress = async () => {
          try {
            const response = await fetch(`${API_BASE_URL}/api/check-progress/${paperId}`);
            if (response.ok) {
              const data = await response.json();
              
              if (data.complete) {
                // Processing is complete, fetch the paper data
                if (checkProgressTimer) {
                  clearInterval(checkProgressTimer);
                }
                finalStageReached = true;
                
                setProcessingProgress(0.95);
                setProcessingMessage('Processing complete, loading results...');
                
                // Short delay to show the completion message
                setTimeout(async () => {
                  try {
                    const paperResponse = await fetch(`${API_BASE_URL}/api/paper/${paperId}`);
                    if (paperResponse.ok) {
                      const paperData = await paperResponse.json();
                      setPaperContent(paperData.content || '');
                      setSummary(paperData.summary || '');
                      setPdfUrl(`${API_BASE_URL}/api/pdf/${paperId}`);
                      
                      setProcessingProgress(1);
                      setProcessingMessage('Paper ready!');
                      
                      setTimeout(() => {
                        setCurrentTab('summary');
                        setIsProcessing(false);
                      }, 1000);
                    } else {
                      throw new Error('Failed to load paper data');
                    }
                  } catch (error) {
                    console.error('Error loading paper data:', error);
                    setIsProcessing(false);
                    setProcessingProgress(0);
                    alert('Error loading paper data. Please try again.');
                  }
                }, 1000);
              } else if (data.progress) {
                // Update progress with real data from backend
                setProcessingProgress(data.progress);
                setProcessingMessage(data.message);
              }
            }
          } catch (error) {
            console.error('Error checking progress:', error);
            // Don't stop the timer on error, just try again
          }
        };
        
        // Simulate progress through each stage, then start checking real progress
        for (let i = 1; i < loadingStages.length; i++) {
          const stage = loadingStages[i];
          
          await new Promise(resolve => {
            setTimeout(() => {
              // Only update if we haven't reached the final stage yet
              if (!finalStageReached) {
                setProcessingProgress(stage.progress);
                setProcessingMessage(stage.message);
              }
              resolve(null);
            }, loadingStages[i - 1].time);
          });
        }
        
        // After simulated stages, start checking real progress
        if (!finalStageReached) {
          setProcessingProgress(0.7);
          setProcessingMessage('Continuing processing...');
          
          // Check real progress every 2 seconds
          checkProgressTimer = setInterval(checkRealProgress, 2000) as NodeJS.Timeout;
          
          // Set a maximum time limit (3 minutes)
          setTimeout(() => {
            if (checkProgressTimer) {
              clearInterval(checkProgressTimer);
              if (!finalStageReached) {
                setIsProcessing(false);
                setProcessingProgress(0);
                alert('Processing is taking longer than expected. The paper may still be processing in the background. Please check again later.');
              }
            }
          }, 180000); // 3 minutes
        }
      }
    } catch (error) {
      console.error('Error processing PDF:', error);
      alert('Failed to process PDF. Please try again.');
      setIsProcessing(false);
      setProcessingProgress(0);
      setProcessingMessage('');
    }
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim()) return;
    
    const newMessages = [
      ...messages,
      { sender: 'user', content: userMessage }
    ];
    
    setMessages(newMessages);
    const currentMessage = userMessage;
    setUserMessage('');
    
    try {
      // Call the backend API to get a response to the question
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: currentMessage,
          paperId: paperId,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      setMessages([
        ...newMessages,
        { 
          sender: 'ai', 
          content: data.response
        }
      ]);
    } catch (error) {
      console.error('Error getting response:', error);
      setMessages([
        ...newMessages,
        { 
          sender: 'ai', 
          content: "I'm sorry, I encountered an error processing your question. Please try again."
        }
      ]);
    }
  };
  
  // Handle keyboard events in the textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If Enter is pressed without holding Cmd (or Ctrl), send the message
    if (e.key === 'Enter' && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
      e.preventDefault(); // Prevent default behavior (new line)
      handleSendMessage();
    }
    // If Cmd+Enter is pressed, allow new line
    // No need to do anything as the default behavior will add a new line
  };

  return (
    <div className="flex flex-col w-full max-w-6xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">AI Research Paper Assistant</CardTitle>
          <CardDescription>Upload an AI research paper to summarize and discuss</CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={currentTab} onValueChange={setCurrentTab}>
            <TabsList className="grid grid-cols-3 mb-6">
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="summary" disabled={!summary}>Summary</TabsTrigger>
              <TabsTrigger value="discuss" disabled={!summary}>Discuss</TabsTrigger>
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
                  <p className="text-sm text-gray-500">Drag and drop or click to upload a PDF</p>
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="application/pdf"
                    onChange={handleFileUpload}
                  />
                  <Button
                    variant="outline"
                    onClick={() => document.getElementById('file-upload')?.click()}
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
                      {isProcessing ? 'Processing...' : 'Process Paper'}
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
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={regenerateSummary}
                      disabled={isRegeneratingSummary}
                      className="flex items-center gap-2"
                    >
                      {isRegeneratingSummary ? (
                        <>
                          <div className="h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                          Regenerating...
                        </>
                      ) : (
                        <>
                          <BookOpenCheck className="h-4 w-4" />
                          Regenerate Summary
                        </>
                      )}
                    </Button>
                  </div>
                  
                  {!summary ? (
                    <div className="flex items-center justify-center h-40 bg-gray-50 rounded-md">
                      <p className="text-gray-500">Loading summary...</p>
                    </div>
                  ) : (
                    <div className="prose prose-blue max-w-none">
                      {summary.split('\n').map((line, i) => {
                        // Handle bold text with asterisks (like **text**)
                        const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                        
                        // Handle headings
                        if (line.startsWith('# ')) {
                          return <h1 key={i} className="text-2xl font-bold mt-6 mb-4 text-blue-800">{line.substring(2)}</h1>;
                        } else if (line.startsWith('## ')) {
                          return <h2 key={i} className="text-xl font-bold mt-5 mb-3 text-blue-700">{line.substring(3)}</h2>;
                        } else if (line.startsWith('### ')) {
                          return <h3 key={i} className="text-lg font-bold mt-4 mb-2 text-blue-600">{line.substring(4)}</h3>;
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
                                <span dangerouslySetInnerHTML={{ __html: content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                              </div>
                            );
                          }
                          return <p key={i} className="my-2 text-gray-700 leading-relaxed">{line}</p>;
                        }
                        // Handle bullet points
                        else if (line.startsWith('* ') || line.startsWith('- ')) {
                          return (
                            <div key={i} className="flex gap-2 my-1 ml-5">
                              <span>•</span>
                              <span dangerouslySetInnerHTML={{ __html: line.substring(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                            </div>
                          );
                        } else if (line.startsWith('  * ') || line.startsWith('  - ')) {
                          return (
                            <div key={i} className="flex gap-2 my-1 ml-10">
                              <span>•</span>
                              <span dangerouslySetInnerHTML={{ __html: line.substring(4).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                            </div>
                          );
                        }
                        // Handle code blocks
                        else if (line.startsWith('```')) {
                          return <div key={i} className="bg-gray-100 p-2 rounded my-2 font-mono text-sm">{line.substring(3)}</div>;
                        } else if (line.startsWith('`') && line.endsWith('`')) {
                          return <code key={i} className="bg-gray-100 px-1 rounded text-sm font-mono">{line.substring(1, line.length - 1)}</code>;
                        }
                        // Handle blockquotes
                        else if (line.startsWith('> ')) {
                          return <blockquote key={i} className="border-l-4 border-gray-300 pl-4 italic my-2" dangerouslySetInnerHTML={{ __html: line.substring(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></blockquote>;
                        }
                        // Handle empty lines
                        else if (line.trim() === '') {
                          return <div key={i} className="my-2"></div>;
                        } 
                        // Handle regular paragraphs
                        else {
                          return <p key={i} className="my-2 text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></p>;
                        }
                      })}
                    </div>
                  )}
                </div>
                
                <div className="flex justify-between">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentTab('upload')}
                    className="flex items-center gap-2"
                  >
                    <UploadCloud className="h-4 w-4" />
                    Upload Different Paper
                  </Button>
                  
                  <Button 
                    onClick={() => setCurrentTab('discuss')}
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
                      src={pdfUrl} 
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
                        <p className="text-sm mt-1">Press Enter to send, Shift+Enter for a new line</p>
                      </div>
                    ) : (
                      messages.map((msg, i) => (
                        <div key={i} className={`${msg.sender === 'user' ? 'ml-auto' : 'mr-auto'} max-w-[85%]`}>
                          <div 
                            className={`p-3 rounded-lg ${
                              msg.sender === 'user' 
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            {msg.content}
                          </div>
                        </div>
                      ))
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