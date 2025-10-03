import React, { useState, useCallback } from 'react';
import { generateMoodBoard } from '../../services/geminiService';
import type { MoodBoardData } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface MoodBoardGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const MoodBoardGenerator: React.FC<MoodBoardGeneratorProps> = ({ tool, onBack }) => {
    const [theme, setTheme] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [moodBoard, setMoodBoard] = useState<MoodBoardData | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!theme.trim()) {
            setError('Por favor, introduce un tema o concepto.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setMoodBoard(null);
        try {
            const data = await generateMoodBoard(theme);
            setMoodBoard(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, [theme]);

    const copyToClipboard = (hex: string) => {
        navigator.clipboard.writeText(hex);
        // Maybe show a small toast notification in the future
    };

    return (
        <div className="max-w-5xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Escribe un tema, idea o concepto (ej: 'marca de café cyberpunk'). La IA generará un collage de 9 imágenes de inspiración y extraerá una paleta de colores coherente.
                </p>
            </div>

            <div className="flex space-x-2">
                <input
                    type="text"
                    value={theme}
                    onChange={(e) => setTheme(e.target.value)}
                    placeholder="Ej: Boda rústica de otoño, diseño interior nórdico..."
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    disabled={isLoading}
                    onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
                >
                    {isLoading ? <Spinner /> : 'Generar Mood Board'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                 {isLoading && (
                    <div className="text-center p-8">
                         <div className="flex justify-center items-center mb-4">
                            <Spinner />
                            <span className="text-lg ml-2">Generando inspiración...</span>
                        </div>
                        <p className="text-gray-400">Esto puede tardar un momento.</p>
                    </div>
                )}
                {moodBoard && (
                     <div className="bg-gray-800/50 p-6 rounded-lg animate-fade-in">
                        <h3 className="text-2xl font-semibold mb-4 text-center">Mood Board para: <span className="text-purple-400">{theme}</span></h3>
                        
                        <div className="grid grid-cols-3 gap-2 md:gap-4 rounded-lg overflow-hidden">
                            {moodBoard.images.map((image, index) => (
                                <div key={index} className="aspect-square">
                                    <img src={image} alt={`Inspiration image ${index + 1}`} className="w-full h-full object-cover" />
                                </div>
                            ))}
                        </div>

                        <div className="mt-6">
                             <h4 className="text-lg font-semibold mb-3 text-center">Paleta de Colores Extraída</h4>
                             <div className="flex justify-center gap-4 flex-wrap">
                                {moodBoard.colors.map((color, index) => (
                                    <div key={index} className="flex flex-col items-center group cursor-pointer" onClick={() => copyToClipboard(color)}>
                                        <div className="w-16 h-16 md:w-20 md:h-20 rounded-full border-2 border-gray-600 transition-transform group-hover:scale-110" style={{ backgroundColor: color }}></div>
                                        <p className="mt-2 text-sm font-mono text-gray-400">{color}</p>
                                    </div>
                                ))}
                             </div>
                             <p className="text-center text-xs text-gray-500 mt-3">Haz clic en un color para copiar su código.</p>
                        </div>
                     </div>
                )}
            </div>
        </div>
    );
};

export default MoodBoardGenerator;