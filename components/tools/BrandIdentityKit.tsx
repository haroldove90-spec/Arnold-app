import React, { useState, useCallback } from 'react';
import { generateBrandKit } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';
import type { KitData } from '../../services/geminiService';

interface BrandIdentityKitProps {
    tool: Tool;
    onBack: () => void;
}

const BrandIdentityKit: React.FC<BrandIdentityKitProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [kitData, setKitData] = useState<KitData | null>(null);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
            setKitData(null);
            setError(null);
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Por favor, sube una imagen de tu logo.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setKitData(null);
        try {
            const response = await generateBrandKit(imageFile);
            setKitData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);

    // FIX: Redefined ColorSwatch using React.FC to correctly handle React-specific props like 'key'
    // and prevent TypeScript errors during JSX transpilation.
    const ColorSwatch: React.FC<{ hex: string; title: string }> = ({ hex, title }) => (
        <div className="text-center">
            <div className="w-16 h-16 rounded-full mx-auto mb-2 border-2 border-gray-700" style={{ backgroundColor: hex }}></div>
            <p className="text-xs font-mono text-gray-400">{hex}</p>
            <p className="text-sm text-gray-200">{title}</p>
        </div>
    );

    return (
        <div className="max-w-6xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Columna de Input */}
                <div className="lg:col-span-1 space-y-4">
                     <div className="bg-gray-800/50 p-6 rounded-lg">
                        <h3 className="text-xl font-semibold mb-4">1. Sube tu Logo</h3>
                         <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg text-center">
                            <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="logo-upload" disabled={isLoading} />
                            <label htmlFor="logo-upload" className="cursor-pointer">
                                {imagePreview ? (
                                    <img src={imagePreview} alt="Logo preview" className="max-h-48 mx-auto rounded" />
                                ) : (
                                    <p className="py-12">Haz clic para subir un logo</p>
                                )}
                            </label>
                        </div>
                    </div>
                     <button
                        onClick={handleGenerate}
                        disabled={isLoading || !imageFile}
                        className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                    >
                        {isLoading && <Spinner />}
                        {isLoading ? 'Generando Kit...' : '2. Generar Kit de Identidad'}
                    </button>
                    {error && <div className="p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
                </div>

                {/* Columna de Resultados */}
                <div className="lg:col-span-2">
                    {isLoading && (
                        <div className="flex items-center justify-center h-full bg-gray-800/50 rounded-lg p-8">
                            <div className="text-center">
                                <Spinner />
                                <p className="mt-4 text-gray-400">Analizando logo y construyendo el kit...</p>
                            </div>
                        </div>
                    )}
                    {!isLoading && !kitData && (
                         <div className="flex items-center justify-center h-full bg-gray-800/50 rounded-lg p-8">
                            <p className="text-gray-500 text-center">Tu kit de identidad de marca aparecerá aquí.</p>
                        </div>
                    )}
                    {kitData && (
                        <div className="space-y-6">
                            {/* Paleta de Colores */}
                            <div className="bg-gray-800/50 p-6 rounded-lg">
                                <h3 className="text-xl font-bold text-purple-400 mb-4">Paleta de Colores</h3>
                                <div className="flex flex-wrap gap-6 justify-center">
                                    {kitData.colors.primary.map((hex, i) => <ColorSwatch key={`p-${i}`} hex={hex} title="Primario" />)}
                                    {kitData.colors.secondary.map((hex, i) => <ColorSwatch key={`s-${i}`} hex={hex} title="Secundario" />)}
                                </div>
                            </div>
                            
                            {/* Tipografía */}
                            <div className="bg-gray-800/50 p-6 rounded-lg">
                                <h3 className="text-xl font-bold text-purple-400 mb-4">Tipografía Sugerida</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                                    <div>
                                        <p className="text-sm text-gray-400">Títulos / Encabezados</p>
                                        <p className="text-2xl font-semibold text-white">{kitData.typography.headlineFont}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-400">Cuerpo de Texto</p>
                                        <p className="text-2xl font-semibold text-white">{kitData.typography.bodyFont}</p>
                                    </div>
                                </div>
                                <p className="text-center text-sm text-gray-300 mt-4 italic bg-gray-900/40 p-3 rounded-md">{kitData.typography.reason}</p>
                            </div>
                            
                            {/* Mockups */}
                             <div className="bg-gray-800/50 p-6 rounded-lg">
                                <h3 className="text-xl font-bold text-purple-400 mb-4">Ejemplos de Aplicación</h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    {kitData.mockups.map((src, i) => (
                                        <img key={i} src={src} alt={`Mockup ${i+1}`} className="w-full aspect-video object-cover rounded-md" />
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default BrandIdentityKit;