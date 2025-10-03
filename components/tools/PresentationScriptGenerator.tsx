import React, { useState, useCallback } from 'react';
import { generateTextWithImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ScriptSection {
    title: string;
    content: string;
}

interface ScriptData {
    script: {
        sections: ScriptSection[];
    };
    rawResponse?: string;
}


interface PresentationScriptGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const PresentationScriptGenerator: React.FC<PresentationScriptGeneratorProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [projectDesc, setProjectDesc] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [scriptData, setScriptData] = useState<ScriptData | null>(null);
    const [copied, setCopied] = useState(false);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Por favor, sube una imagen de tu diseño.');
            return;
        }
        if (!projectDesc.trim()) {
            setError('Por favor, describe el proyecto o los objetivos del cliente.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setScriptData(null);
        try {
            const prompt = `Actúa como un director creativo presentando un diseño a un cliente. El objetivo del proyecto es: "${projectDesc}". Basado en la imagen del diseño proporcionada, escribe un guion de presentación convincente. Responde en JSON con la siguiente estructura: { "script": { "sections": [{ "title": "Nombre de la Sección", "content": "Texto del guion para esta sección..." }] } }. Las secciones deben ser lógicas, como 'Introducción', 'Análisis del Diseño', 'Decisiones Creativas', y 'Conclusión'.`;
            const response = await generateTextWithImage(prompt, imageFile, true);
            setScriptData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile, projectDesc]);
    
    const handleCopy = () => {
        if (scriptData?.script?.sections) {
            const fullScript = scriptData.script.sections
                .map(section => `${section.title}\n\n${section.content}`)
                .join('\n\n---\n\n');
            navigator.clipboard.writeText(fullScript);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } else if (scriptData?.rawResponse) {
            navigator.clipboard.writeText(scriptData.rawResponse);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
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
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube tu diseño final y describe brevemente los objetivos del cliente. La IA generará un guion profesional para que presentes tu trabajo con confianza.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg space-y-4">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg text-center">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload-script" disabled={isLoading} />
                    <label htmlFor="image-upload-script" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-40 mx-auto rounded-lg" />
                        ) : (
                            <p>Haz clic para subir el diseño final</p>
                        )}
                    </label>
                </div>
                 <textarea
                    value={projectDesc}
                    onChange={(e) => setProjectDesc(e.target.value)}
                    placeholder="Describe los objetivos del cliente (ej: 'Crear un logo para una cafetería que sea moderno y acogedor...')"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={3}
                    disabled={isLoading}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile || !projectDesc.trim()}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Guion...' : 'Generar Guion de Presentación'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">La IA está preparando tu presentación...</p>}
                {scriptData && (
                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-semibold">Guion de Presentación</h3>
                            <button onClick={handleCopy} className="bg-gray-700 text-sm text-gray-200 px-3 py-1 rounded-md hover:bg-gray-600 transition-colors">
                                {copied ? '¡Copiado!' : 'Copiar Guion Completo'}
                            </button>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg space-y-2">
                            {scriptData.script?.sections ? (
                                scriptData.script.sections.map((section, index) => (
                                    <details key={index} className="bg-gray-900/50 rounded-lg" open={index === 0}>
                                        <summary className="font-semibold text-purple-400 p-4 cursor-pointer">
                                            {section.title}
                                        </summary>
                                        <div className="p-4 border-t border-gray-700">
                                            <p className="whitespace-pre-wrap font-sans text-gray-300">{section.content}</p>
                                        </div>
                                    </details>
                                ))
                            ) : (
                                <div>
                                    <p className="text-yellow-400 text-sm mb-4">La IA no devolvió un formato estructurado. Mostrando respuesta en bruto:</p>
                                    <pre className="whitespace-pre-wrap text-left bg-gray-900 p-4 rounded font-sans text-gray-300">{scriptData.rawResponse}</pre>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PresentationScriptGenerator;