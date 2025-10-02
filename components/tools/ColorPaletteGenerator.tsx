
import React, { useState, useCallback } from 'react';
import { generateColorPalette } from '../../services/geminiService';
import type { Color } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ColorPaletteGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const ColorPaletteGenerator: React.FC<ColorPaletteGeneratorProps> = ({ tool, onBack }) => {
    const [theme, setTheme] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [palette, setPalette] = useState<Color[] | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!theme.trim()) {
            setError('Please enter a theme or description.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setPalette(null);
        try {
            const colors = await generateColorPalette(theme);
            setPalette(colors);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [theme]);

    const copyToClipboard = (hex: string) => {
        navigator.clipboard.writeText(hex);
    };

    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Escribe un tema, una idea o una palabra clave (ej: 'atardecer en el desierto'). La IA generará una paleta de 5 colores armoniosos con sus nombres y códigos hexadecimales.
                </p>
            </div>

            <div className="flex space-x-2">
                <input
                    type="text"
                    value={theme}
                    onChange={(e) => setTheme(e.target.value)}
                    placeholder="Ej: a calm and relaxing spa"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    disabled={isLoading}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
                >
                    {isLoading ? <Spinner /> : 'Generar'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                 {isLoading && (
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {Array.from({ length: 5 }).map((_, i) => (
                            <div key={i} className="h-40 bg-gray-800 rounded-lg animate-pulse"></div>
                        ))}
                    </div>
                )}
                {palette && (
                     <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Paleta de Colores Generada</h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                            {palette.map((color, index) => (
                                <div key={index} className="group cursor-pointer" onClick={() => copyToClipboard(color.hex)}>
                                    <div className="h-32 rounded-lg" style={{ backgroundColor: color.hex }}></div>
                                    <div className="p-2 text-center">
                                        <p className="font-medium text-sm text-gray-200">{color.name}</p>
                                        <p className="text-xs text-gray-400 group-hover:text-purple-400 transition-colors">{color.hex}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <p className="text-center text-sm text-gray-500 mt-4">Haz clic en un color para copiar su código hexadecimal.</p>
                     </div>
                )}
            </div>
        </div>
    );
};

export default ColorPaletteGenerator;
