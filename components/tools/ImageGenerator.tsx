import React, { useState, useCallback } from 'react';
import { generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ImageGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const ImageGenerator: React.FC<ImageGeneratorProps> = ({ tool, onBack }) => {
    const [prompt, setPrompt] = useState('');
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedImages, setGeneratedImages] = useState<string[]>([]);

    const handleGenerate = useCallback(async () => {
        if (!prompt.trim()) {
            setError('Please enter a prompt.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedImages([]);
        try {
            const imageUrls = await generateImage(prompt, 2, aspectRatio);
            setGeneratedImages(imageUrls);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [prompt, aspectRatio]);

    const downloadImage = (imageUrl: string, index: number) => {
        const link = document.createElement('a');
        link.href = imageUrl;
        // Sanitize prompt for filename
        const safePrompt = prompt.slice(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.download = `ai_design_${safePrompt}_${index + 1}.jpeg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Escribe una descripción detallada, elige el formato y la IA generará dos imágenes únicas basadas en tu texto.
                </p>
            </div>

            <div className="space-y-4">
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Ej: A futuristic logo for a tech company, minimalist, blue and silver..."
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={3}
                    disabled={isLoading}
                />
                <div className="flex flex-col sm:flex-row gap-4">
                    <div className="flex-grow">
                         <label htmlFor="aspect-ratio" className="block text-sm font-medium text-gray-400 mb-1">Formato</label>
                         <select
                            id="aspect-ratio"
                            value={aspectRatio}
                            onChange={(e) => setAspectRatio(e.target.value)}
                            className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                            disabled={isLoading}
                        >
                            <option value="1:1">Cuadrado (1:1)</option>
                            <option value="3:4">Retrato (3:4)</option>
                            <option value="4:3">Paisaje (4:3)</option>
                            <option value="16:9">Panorámico (16:9)</option>
                            <option value="9:16">Historia (9:16)</option>
                        </select>
                    </div>
                    <button
                        onClick={handleGenerate}
                        disabled={isLoading}
                        className="w-full sm:w-auto self-end flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                    >
                        {isLoading && <Spinner />}
                        {isLoading ? 'Generando...' : 'Generar Imágenes'}
                    </button>
                </div>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="aspect-square w-full bg-gray-800 rounded-lg flex items-center justify-center animate-pulse">
                            <p className="text-gray-500">Generando imagen 1...</p>
                        </div>
                        <div className="aspect-square w-full bg-gray-800 rounded-lg flex items-center justify-center animate-pulse">
                            <p className="text-gray-500">Generando imagen 2...</p>
                        </div>
                    </div>
                )}
                {generatedImages.length > 0 && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Resultados</h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {generatedImages.map((image, index) => (
                                <div key={index} className="group relative">
                                    <img src={image} alt={`Generated design ${index + 1}`} className="w-full aspect-auto object-cover rounded-lg shadow-lg" />
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg">
                                        <button onClick={() => downloadImage(image, index)} className="text-white bg-purple-600/80 hover:bg-purple-600 px-4 py-2 rounded-lg font-semibold flex items-center gap-2">
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                                            Descargar
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageGenerator;