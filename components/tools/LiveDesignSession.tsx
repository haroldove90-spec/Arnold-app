import React, { useState, useCallback, useEffect, useRef } from 'react';
import { generateDetailedBrief, generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface LiveDesignSessionProps {
    tool: Tool;
    onBack: () => void;
}

const LiveDesignSession: React.FC<LiveDesignSessionProps> = ({ tool, onBack }) => {
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [detailedBrief, setDetailedBrief] = useState<string | null>(null);
    const [displayedBrief, setDisplayedBrief] = useState('');
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [isGeneratingImage, setIsGeneratingImage] = useState(false);

    const briefContainerRef = useRef<HTMLDivElement>(null);

    // Effect for the "typewriter" animation
    useEffect(() => {
        if (detailedBrief) {
            let i = 0;
            setDisplayedBrief(''); // Reset displayed brief
            const interval = setInterval(() => {
                setDisplayedBrief(prev => prev + detailedBrief[i]);
                i++;
                if (briefContainerRef.current) {
                    briefContainerRef.current.scrollTop = briefContainerRef.current.scrollHeight;
                }
                if (i === detailedBrief.length) {
                    clearInterval(interval);
                    // Once text is finished, generate image
                    const generate = async () => {
                        setIsGeneratingImage(true);
                        try {
                            const imageUrls = await generateImage(detailedBrief, 1, '4:3');
                            setGeneratedImage(imageUrls[0]);
                        } catch (err) {
                             setError(err instanceof Error ? err.message : 'Ocurrió un error al generar la imagen.');
                        } finally {
                            setIsGeneratingImage(false);
                        }
                    };
                    generate();
                }
            }, 20); // Adjust speed of typing here
            return () => clearInterval(interval);
        }
    }, [detailedBrief]);

    const handleStartSession = useCallback(async () => {
        if (!prompt.trim()) {
            setError('Por favor, introduce una idea para empezar.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setDetailedBrief(null);
        setDisplayedBrief('');
        setGeneratedImage(null);
        setIsGeneratingImage(false);

        try {
            const brief = await generateDetailedBrief(prompt);
            setDetailedBrief(brief);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false); // Loading is for brief generation, image generation has its own state
        }
    }, [prompt]);

    const handleReset = () => {
        setPrompt('');
        setDetailedBrief(null);
        setDisplayedBrief('');
        setGeneratedImage(null);
        setError(null);
    };

    return (
        <div className="max-w-6xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            {!detailedBrief ? (
                <div className="max-w-2xl mx-auto">
                    <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                        <p className="text-sm text-gray-300">
                            <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Escribe una idea inicial. La IA la expandirá en un brief creativo detallado y luego generará una imagen basada en ese brief mejorado.
                        </p>
                    </div>
                    <div className="space-y-4">
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Ej: Un logo para una marca de jabones ecológicos"
                            className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                            rows={3}
                            disabled={isLoading}
                        />
                        <button
                            onClick={handleStartSession}
                            disabled={isLoading}
                            className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                        >
                            {isLoading && <Spinner />}
                            {isLoading ? 'Desarrollando Idea...' : 'Iniciar Sesión de Diseño'}
                        </button>
                    </div>
                </div>
            ) : (
                <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Panel del Brief */}
                        <div className="bg-gray-800/50 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-2 text-purple-400">Brief Creativo de la IA</h3>
                            <div ref={briefContainerRef} className="h-96 overflow-y-auto bg-gray-900 p-3 rounded-md text-gray-300 whitespace-pre-wrap font-mono text-sm">
                                {displayedBrief}
                                {!isGeneratingImage && <span className="animate-ping">_</span>}
                            </div>
                        </div>

                        {/* Panel de la Imagen */}
                        <div className="bg-gray-800/50 p-4 rounded-lg flex flex-col items-center justify-center">
                            <h3 className="text-lg font-semibold mb-2 text-purple-400">Lienzo Visual</h3>
                            <div className="w-full aspect-video bg-gray-900 rounded-md flex items-center justify-center">
                                {isGeneratingImage && (
                                    <div className="text-center">
                                        <Spinner />
                                        <p className="text-sm text-gray-400 mt-2">Generando imagen...</p>
                                    </div>
                                )}
                                {generatedImage && <img src={generatedImage} alt="Resultado final" className="w-full h-full object-cover rounded-md" />}
                                {!generatedImage && !isGeneratingImage && <p className="text-gray-500">La imagen aparecerá aquí</p>}
                            </div>
                        </div>
                    </div>
                    <div className="text-center">
                        <button onClick={handleReset} className="bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-6 rounded-lg transition-colors">
                            Iniciar Nueva Sesión
                        </button>
                    </div>
                </div>
            )}

            {error && <div className="mt-6 max-w-2xl mx-auto p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
        </div>
    );
};

export default LiveDesignSession;
