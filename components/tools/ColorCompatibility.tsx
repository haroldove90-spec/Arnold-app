
import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';
import { Type } from '@google/genai';

interface ColorCompatibilityProps {
    tool: Tool;
    onBack: () => void;
}

interface AnalysisResult {
    ratio: number;
    aa_normal: boolean;
    aaa_normal: boolean;
    aa_large: boolean;
    aaa_large: boolean;
    feedback: string;
}

const ColorCompatibility: React.FC<ColorCompatibilityProps> = ({ tool, onBack }) => {
    const [color1, setColor1] = useState('#ffffff');
    const [color2, setColor2] = useState('#000000');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);

    const handleGenerate = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `Analyze the color contrast between foreground color ${color1} and background color ${color2}. Is it compliant with WCAG AA and AAA standards for normal and large text? Provide the contrast ratio and a brief explanation.`;
            const schema = {
                type: Type.OBJECT,
                properties: {
                    ratio: { type: Type.NUMBER, description: "The contrast ratio." },
                    aa_normal: { type: Type.BOOLEAN, description: "Passes WCAG AA for normal text." },
                    aaa_normal: { type: Type.BOOLEAN, description: "Passes WCAG AAA for normal text." },
                    aa_large: { type: Type.BOOLEAN, description: "Passes WCAG AA for large text." },
                    aaa_large: { type: Type.BOOLEAN, description: "Passes WCAG AAA for large text." },
                    feedback: { type: Type.STRING, description: "A brief summary of the results." }
                },
                required: ["ratio", "aa_normal", "aaa_normal", "aa_large", "aaa_large", "feedback"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [color1, color2]);

    const StatusBadge: React.FC<{pass: boolean}> = ({ pass }) => (
        <span className={`px-2 py-1 text-xs font-semibold rounded-full ${pass ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}`}>
            {pass ? 'PASS' : 'FAIL'}
        </span>
    );

    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Selecciona dos colores (uno para el texto y otro para el fondo) usando los selectores. La IA analizará el contraste y te dirá si cumple con las normas de accesibilidad web (WCAG).
                </p>
            </div>

            <div className="space-y-4 md:space-y-0 md:flex md:space-x-4 items-center">
                <div className="flex-1 space-y-2">
                    <label htmlFor="color1" className="text-sm">Texto / Primer plano</label>
                    <input id="color1" type="color" value={color1} onChange={(e) => setColor1(e.target.value)} className="w-full h-12 p-1 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                </div>
                <div className="flex-1 space-y-2">
                    <label htmlFor="color2" className="text-sm">Fondo</label>
                    <input id="color2" type="color" value={color2} onChange={(e) => setColor2(e.target.value)} className="w-full h-12 p-1 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full md:w-auto self-end flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
                >
                    {isLoading ? <Spinner /> : 'Analizar'}
                </button>
            </div>
            <div className="mt-4 flex justify-around text-center">
                <p className="font-mono">{color1}</p>
                <p className="font-mono">{color2}</p>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center">Analizando colores...</p>}
                {result && (
                    <div className="bg-gray-800 p-6 rounded-lg">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xl font-semibold">Ratio de Contraste</h3>
                            <p className="text-3xl font-bold text-purple-400">{result.ratio.toFixed(2)}:1</p>
                        </div>
                         <div className="mt-4 p-8 rounded" style={{ backgroundColor: color2, color: color1 }}>
                            <h4 className="text-2xl font-bold">Texto grande de ejemplo</h4>
                            <p className="mt-2">Texto normal de ejemplo. La legibilidad es clave para un diseño accesible e inclusivo.</p>
                        </div>
                        <div className="mt-6 space-y-3">
                            <div className="flex justify-between items-center"><p>WCAG AA (Texto normal)</p> <StatusBadge pass={result.aa_normal} /></div>
                            <div className="flex justify-between items-center"><p>WCAG AAA (Texto normal)</p> <StatusBadge pass={result.aaa_normal} /></div>
                            <div className="flex justify-between items-center"><p>WCAG AA (Texto grande)</p> <StatusBadge pass={result.aa_large} /></div>
                            <div className="flex justify-between items-center"><p>WCAG AAA (Texto grande)</p> <StatusBadge pass={result.aaa_large} /></div>
                        </div>
                         <p className="mt-6 text-center text-gray-400 italic">{result.feedback}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ColorCompatibility;
