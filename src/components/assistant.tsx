"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { UploadCloud, BookOpen, MessageSquare, FileAudio, FileText } from 'lucide-react';

// API base URL - change this if your Flask server runs on a different port
const API_BASE_URL = 'http://127.0.0.1:5000';


const Assistant = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTab, setCurrentTab] = useState('upload');
  const [paperContent, setPaperContent] = useState('');
  const [summary, setSummary] = useState('');
  const [messages, setMessages] = useState<Array<{sender: string, content: string}>>([]);
  const [userMessage, setUserMessage] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);

  // Add this to your component
useEffect(() => {
    // Test the backend connection when component mounts
    fetch(`${API_BASE_URL}/api/test`)
      .then(response => response.json())
      .then(data => {
        console.log("Backend connection successful:", data);
      })
      .catch(error => {
        console.error("Backend connection failed:", error);
      });
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (uploadedFile && uploadedFile.type === 'application/pdf') {
      setFile(uploadedFile);
    } else {
      alert('Please upload a PDF file');
    }
  };

  const processPaper = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    
    try {
      // Create form data for file upload
      const formData = new FormData();
      formData.append('file', file);
      
      // Call the backend API to process the PDF
      const response = await fetch(`${API_BASE_URL}/api/process-pdf`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setPaperContent(data.extractedText);
        setSummary(data.summary);
        setCurrentTab('summary');
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error processing PDF:', error);
      alert('Failed to process PDF. Please try again.');
    } finally {
      setIsProcessing(false);
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
          question: currentMessage,
          paperContent: paperContent,
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
          content: data.answer
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

  const generateAudio = async () => {
    setIsGeneratingAudio(true);
    
    try {
      // Call the backend API to generate audio
      const response = await fetch(`${API_BASE_URL}/api/generate-audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: summary,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setAudioUrl(`${API_BASE_URL}${data.audioUrl}`);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error generating audio:', error);
      alert('Failed to generate audio. Please try again.');
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  return (
    <div className="flex flex-col w-full max-w-4xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">AI Research Paper Assistant</CardTitle>
          <CardDescription>Upload an AI research paper to summarize, discuss, and generate audio transcripts</CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={currentTab} onValueChange={setCurrentTab}>
            <TabsList className="grid grid-cols-4 mb-6">
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="summary" disabled={!summary}>Summary</TabsTrigger>
              <TabsTrigger value="discuss" disabled={!summary}>Discuss</TabsTrigger>
              <TabsTrigger value="audio" disabled={!summary}>Audio</TabsTrigger>
            </TabsList>
            
            <TabsContent value="upload" className="space-y-4">
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
                    {summary.split('\n').map((line, i) => (
                      line.startsWith('# ') ? (
                        <h1 key={i} className="text-xl font-bold mt-4">{line.substring(2)}</h1>
                      ) : line.startsWith('## ') ? (
                        <h2 key={i} className="text-lg font-bold mt-3">{line.substring(3)}</h2>
                      ) : (
                        <p key={i} className="my-2">{line}</p>
                      )
                    ))}
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="discuss" className="space-y-4">
              <div className="border rounded-lg p-4 h-80 overflow-y-auto flex flex-col space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center text-gray-500 my-auto">
                    <MessageSquare className="h-12 w-12 mx-auto opacity-30" />
                    <p className="mt-2">Ask questions about the paper</p>
                  </div>
                ) : (
                  messages.map((msg, i) => (
                    <div key={i} className={`max-w-3/4 ${msg.sender === 'user' ? 'ml-auto' : 'mr-auto'}`}>
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
              
              <div className="flex gap-2">
                <Textarea
                  placeholder="Ask about the paper..."
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  className="flex-1"
                />
                <Button 
                  onClick={handleSendMessage}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  Send
                </Button>
              </div>
            </TabsContent>
            
            <TabsContent value="audio" className="space-y-4">
              <div className="text-center space-y-6 py-8">
                <FileAudio className="h-16 w-16 mx-auto text-gray-400" />
                <div>
                  <Button 
                    onClick={generateAudio}
                    disabled={isGeneratingAudio}
                    className="bg-blue-600 hover:bg-blue-700"
                    size="lg"
                  >
                    {isGeneratingAudio ? 'Generating Audio...' : 'Generate Audio Transcript'}
                  </Button>
                </div>
                
                {audioUrl && (
                  <div className="mt-8 border rounded-lg p-4">
                    <h3 className="font-medium mb-2">Audio Transcript:</h3>
                    <div className="bg-gray-100 p-3 rounded">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-500">paper_summary_audio.mp3</span>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => window.open(audioUrl, '_blank')}
                        >
                          Download
                        </Button>
                      </div>
                      <div className="mt-2">
                        <audio controls className="w-full">
                          <source src={audioUrl} type="audio/mpeg" />
                          Your browser does not support the audio element.
                        </audio>
                      </div>
                    </div>
                  </div>
                )}
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