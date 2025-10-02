
import React, { useState, useCallback } from 'react';
import { generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface VisualBrainstormingProps {
    tool: Tool;
    onBack: () => void;
}

const VisualBrainstorming: React.FC<VisualBrainstormingProps> = ({ tool, onBack }) => {
    const [prompt, setPrompt] = useState('');
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
            const imageUrls = await generateImage(prompt, 4);
            setGeneratedImages(imageUrls);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [prompt]);
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Introduce un concepto o idea visual. La IA generará rápidamente cuatro imágenes diferentes basadas en tu idea, ayudándote a explorar distintas direcciones creativas.
                </p>
            </div>

            <div className="space-y-4">
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Ej: A mascot for a coffee brand, friendly monster, vector illustration..."
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={3}
                    disabled={isLoading}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Variantes...' : 'Generar 4 Variantes'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Generando variantes... por favor espere.</p>}
                {generatedImages.length > 0 && (
                     <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Variantes Generadas</h3>
                        <div className="grid grid-cols-2 gap-4">
                            {generatedImages.map((image, index) => (
                                <img key={index} src={image} alt={`Variant ${index + 1}`} className="w-full aspect-square object-cover rounded-lg shadow-lg" />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default VisualBrainstorming;
