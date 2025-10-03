import React, { useState, useCallback, useMemo } from 'react';
import { generateTextWithImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface SemanticSearchProps {
    tool: Tool;
    onBack: () => void;
}

interface Asset {
    id: number;
    file: File;
    preview: string;
    description: string;
}

const SemanticSearch: React.FC<SemanticSearchProps> = ({ tool, onBack }) => {
    const [assets, setAssets] = useState<Asset[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [processingStatus, setProcessingStatus] = useState('');

    const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        setIsLoading(true);
        setError(null);
        setAssets([]); // Clear previous assets
        setSearchQuery('');

        // FIX: Explicitly typing `file` as `File` resolves incorrect type inference from `Array.from(files)`.
        const newAssets: Omit<Asset, 'description'>[] = Array.from(files).map((file: File, index) => ({
            id: Date.now() + index,
            file,
            preview: URL.createObjectURL(file),
        }));

        const describedAssets: Asset[] = [];

        for (let i = 0; i < newAssets.length; i++) {
            setProcessingStatus(`Analizando imagen ${i + 1} de ${newAssets.length}...`);
            try {
                const prompt = "Describe this image in a concise sentence for semantic search purposes. Include main objects, colors, and style.";
                const description = await generateTextWithImage(prompt, newAssets[i].file);
                describedAssets.push({ ...newAssets[i], description });
            } catch (err) {
                console.error(`Failed to get description for ${newAssets[i].file.name}`, err);
                // Add asset even if description fails, so it's not lost
                describedAssets.push({ ...newAssets[i], description: 'Error: No se pudo generar la descripción.' });
            }
        }
        
        setAssets(describedAssets);
        setIsLoading(false);
        setProcessingStatus('');

    }, []);

    const filteredAssets = useMemo(() => {
        if (!searchQuery.trim()) {
            return assets;
        }
        const lowerCaseQuery = searchQuery.toLowerCase();
        return assets.filter(asset => asset.description.toLowerCase().includes(lowerCaseQuery));
    }, [searchQuery, assets]);

    return (
        <div className="max-w-6xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Carga tus imágenes. La IA las analizará y describirá. Luego, usa la barra de búsqueda para encontrar lo que necesites con lenguaje natural (ej: "logo azul con una montaña").
                </p>
            </div>

            {assets.length === 0 && !isLoading && (
                 <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                    <div className="w-full p-8 border-2 border-dashed border-gray-600 rounded-lg">
                        <input type="file" multiple accept="image/*" onChange={handleFileChange} className="hidden" id="asset-upload" disabled={isLoading} />
                        <label htmlFor="asset-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                            <p className="font-semibold text-lg">Haz clic o arrastra tus imágenes aquí</p>
                            <p className="text-sm text-gray-400 mt-1">Sube tu librería de assets para empezar a buscar</p>
                        </label>
                    </div>
                </div>
            )}
            
            {isLoading && (
                <div className="text-center p-8">
                    <div className="flex justify-center items-center mb-4">
                        <Spinner />
                        <span className="text-lg ml-2">Procesando...</span>
                    </div>
                    <p className="text-gray-400">{processingStatus}</p>
                </div>
            )}

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}

            {assets.length > 0 && !isLoading && (
                <div className="space-y-6">
                    <div className="sticky top-16 md:top-20 z-20 bg-gray-900/80 backdrop-blur-sm py-4 px-2 -mx-2 rounded-b-lg">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Buscar en tus assets... (ej: 'logo minimalista en blanco y negro')"
                            className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                        />
                    </div>
                    
                     <div className="flex justify-between items-center px-2">
                        <p className="text-gray-400 text-sm">{filteredAssets.length} de {assets.length} assets mostrados.</p>
                        <label htmlFor="asset-upload-new" className="text-sm text-purple-400 hover:text-purple-300 cursor-pointer">
                            Cargar más
                            <input type="file" multiple accept="image/*" onChange={handleFileChange} className="hidden" id="asset-upload-new" />
                        </label>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        {filteredAssets.map(asset => (
                            <div key={asset.id} className="group relative aspect-square">
                                <img src={asset.preview} alt={asset.file.name} className="w-full h-full object-cover rounded-lg" />
                                <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity p-2 text-xs text-white overflow-hidden rounded-lg">
                                    {asset.description}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
};

export default SemanticSearch;