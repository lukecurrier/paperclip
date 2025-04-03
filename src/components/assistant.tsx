"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { UploadCloud, BookOpen, MessageSquare, FileText } from 'lucide-react';

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
        setProcessingProgress(0.1);
        setProcessingMessage('Starting PDF processing...');
        
        // Create form data for file upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('paperId', paperId);
        
        // Setup event source for progress updates
        const eventSource = new EventSource(`${API_BASE_URL}/api/process-progress/${paperId}`);
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          setProcessingProgress(data.progress);
          setProcessingMessage(data.message);
          
          if (data.progress >= 1) {
            eventSource.close();
          }
        };
        
        eventSource.onerror = () => {
          eventSource.close();
        };
        
        // Call the backend API to process the PDF
        const response = await fetch(`${API_BASE_URL}/api/process-pdf`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          eventSource.close();
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
          setPaperContent(data.extractedText);
          setSummary(data.summary);
          setPdfUrl(`${API_BASE_URL}/api/pdf/${paperId}`);
          setCurrentTab('summary');
        } else {
          alert(`Error: ${data.error}`);
        }
      }
    } catch (error) {
      console.error('Error processing PDF:', error);
      alert('Failed to process PDF. Please try again.');
    } finally {
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingProgress(0);
        setProcessingMessage('');
      }, 1000);
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
              <div className="space-y-4">
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h3 className="font-medium text-sm mb-2 text-gray-500">Original Paper Content:</h3>
                  <div className="max-h-40 overflow-y-auto p-2 bg-white rounded border">
                    {paperContent}
                  </div>
                </div>
                
                <div className="border rounded-lg p-4">
                  <h3 className="font-medium text-sm mb-2 flex items-center gap-2">
                    <BookOpen className="h-4 w-4" />
                    <span>Paper Summary:</span>
                  </h3>
                  <div className="prose max-w-none">
                    {summary && summary.split('\n').map((line, i) => (
                      line.startsWith('# ') ? (
                        <h1 key={i} className="text-xl font-bold mt-4">{line.substring(2)}</h1>
                      ) : line.startsWith('## ') ? (
                        <h2 key={i} className="text-lg font-bold mt-3">{line.substring(3)}</h2>
                      ) : (
                        <p key={i} className="my-2">{line}</p>
                      )
                    ))}
                    {!summary && (
                      <p className="text-gray-500">Loading summary...</p>
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="discuss" className="space-y-4">
              <div className="flex gap-4 h-[700px]">
                {/* PDF Viewer */}
                <div className="w-2/5 border rounded-lg overflow-hidden">
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
                        <p className="text-sm mt-1">Press Enter to send, Cmd+Enter for a new line</p>
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
        
        <CardFooter className="flex justify-between">
          <div className="text-sm text-gray-500">
            {file ? `Current file: ${file.name}` : 'No file selected'}
          </div>
          <div>
            {currentTab !== 'upload' && (
              <Button variant="outline" onClick={() => setCurrentTab('upload')}>
                Upload New Paper
              </Button>
            )}
          </div>
        </CardFooter>
      </Card>
    </div>
  );
};

export default Assistant;