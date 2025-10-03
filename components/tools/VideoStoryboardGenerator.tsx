import React, { useState, useCallback } from 'react';
import { generateStoryboard } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface VideoStoryboardGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const SHOT_TYPES = ["Plano General", "Plano Medio", "Primer Plano", "Plano de Cierre"];

const VideoStoryboardGenerator: React.FC<VideoStoryboardGeneratorProps> = ({ tool, onBack }) => {
    const [scene, setScene] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [storyboard, setStoryboard] = useState<string[]>([]);

    const handleGenerate = useCallback(async () => {
        if (!scene.trim()) {
            setError('Por favor, describe la escena.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setStoryboard([]);
        try {
            const imageUrls = await generateStoryboard(scene);
            if (imageUrls.length < 4) {
                 throw new Error("La IA no generó suficientes paneles para el storyboard.");
            }
            setStoryboard(imageUrls);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, [scene]);

    return (
        <div className="max-w-6xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe una escena o un momento de un guion. La IA generará 4 imágenes que representan los planos clave de tu historia, perfectas para la pre-producción.
                </p>
            </div>

            <div className="space-y-4">
                <textarea
                    value={scene}
                    onChange={(e) => setScene(e.target.value)}
                    placeholder="Ej: Un detective camina por un callejón lluvioso de noche, iluminado por un único letrero de neón que parpadea."
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
                    {isLoading ? 'Generando Storyboard...' : 'Generar Storyboard'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {SHOT_TYPES.map((title) => (
                             <div key={title} className="bg-gray-800/50 p-2 rounded-lg">
                                <div className="aspect-video w-full bg-gray-800 rounded-md flex items-center justify-center animate-pulse">
                                    <p className="text-gray-500 text-sm">Generando...</p>
                                </div>
                                <h4 className="font-semibold text-center mt-2 text-purple-400">{title}</h4>
                            </div>
                        ))}
                    </div>
                )}
                {storyboard.length > 0 && (
                     <div>
                        <h3 className="text-2xl font-semibold mb-4 text-center">Storyboard para: <span className="text-purple-400/80">{`"${scene.substring(0, 40)}..."`}</span></h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {storyboard.map((image, index) => (
                                <div key={index} className="bg-gray-800/50 p-2 rounded-lg">
                                    <img src={image} alt={`Storyboard panel ${index + 1}`} className="w-full aspect-video object-cover rounded-md shadow-lg" />
                                    <h4 className="font-semibold text-center mt-2 text-purple-400">{SHOT_TYPES[index] || `Panel ${index + 1}`}</h4>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default VideoStoryboardGenerator;