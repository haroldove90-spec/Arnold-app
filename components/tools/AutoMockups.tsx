
import React, { useState, useCallback } from 'react';
import { generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface AutoMockupsProps {
    tool: Tool;
    onBack: () => void;
}

const AutoMockups: React.FC<AutoMockupsProps> = ({ tool, onBack }) => {
    const [design, setDesign] = useState('');
    const [context, setContext] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!design.trim() || !context.trim()) {
            setError('Please provide both a design description and a context.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedImage(null);
        try {
            const prompt = `Create a realistic mockup image. The design is: "${design}". Display this design on the following context: "${context}".`;
            const imageUrls = await generateImage(prompt, 1);
            setGeneratedImage(imageUrls[0]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [design, context]);
    
    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

             <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe tu diseño y el contexto donde quieres verlo. La IA creará una imagen realista de tu diseño aplicado en ese entorno, como un logo en una taza o una app en un teléfono.
                </p>
            </div>

            <div className="space-y-4">
                 <textarea
                    value={design}
                    onChange={(e) => setDesign(e.target.value)}
                    placeholder="Describe the design (e.g., a minimalist logo of a mountain in blue)"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={2}
                    disabled={isLoading}
                />
                 <textarea
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    placeholder="Describe the mockup context (e.g., on a coffee cup on a wooden table)"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={2}
                    disabled={isLoading}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Mockup...' : 'Generar Mockup'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && (
                    <div className="aspect-square w-full bg-gray-800 rounded-lg flex items-center justify-center animate-pulse">
                        <p className="text-gray-500">Generando mockup...</p>
                    </div>
                )}
                {generatedImage && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Mockup Generado</h3>
                        <img src={generatedImage} alt="Generated mockup" className="w-full aspect-square object-cover rounded-lg shadow-lg" />
                    </div>
                )}
            </div>
        </div>
    );
};

export default AutoMockups;
