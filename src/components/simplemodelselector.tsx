"use client";

import React, { useEffect, useState } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8000';
console.log("API_BASE_URL is:", API_BASE_URL);

interface Model {
  id: string;
  name: string;
  description: string;
  provider: string;
}

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelId: string) => void;
}

const SimpleModelSelector: React.FC<ModelSelectorProps> = ({ selectedModel, onModelChange }) => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/models`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.status}`);
        }
        
        const data = await response.json();
        setModels(data);
      } catch (error) {
        console.error('Error fetching models:', error);
        setError('Failed to load models');
        setModels([
          {
            id: "llama-3.1-8b",
            name: "Llama 3.1 8B",
            description: "Larger model with better reasoning",
            provider: "openai"
          },
          {
            id: "llama-3.2-1b",
            name: "Llama 3.2 1B",
            description: "Smaller, faster model",
            provider: "huggingface"
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onModelChange(e.target.value);
  };

  if (loading) {
    return (
      <div className="flex items-center text-sm text-gray-500">
        <span className="mr-2">Loading models...</span>
        <div className="h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-500">Model:</span>
      <select 
        value={selectedModel} 
        onChange={handleModelChange}
        className="h-9 px-3 py-1 rounded border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default SimpleModelSelector;