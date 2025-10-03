
import React from 'react';
import type { Tool } from '../../types';

interface PlaceholderToolProps {
    tool: Tool;
    onBack: () => void;
}

const PlaceholderTool: React.FC<PlaceholderToolProps> = ({ tool, onBack }) => {
    return (
        <div className="max-w-3xl mx-auto text-center">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            
            <div className="bg-gray-800/50 p-8 rounded-lg">
                <div className="text-purple-400 inline-block mb-4">
                    {/* The icon type is now React.ReactElement, so we can safely clone it to apply new props. */}
                    {React.isValidElement(tool.icon) ? React.cloneElement(tool.icon, { className: "h-12 w-12" }) : tool.icon}
                </div>
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2 mb-6">{tool.description}</p>
                <div className="bg-yellow-500/20 text-yellow-300 px-4 py-2 rounded-full inline-block">
                    <p className="font-semibold">Próximamente</p>
                </div>
                 <p className="text-sm text-gray-400 mt-6">
                    Esta herramienta aún no está disponible, pero estamos trabajando en ella. ¡Vuelve a consultar pronto!
                </p>
            </div>
        </div>
    );
};

export default PlaceholderTool;