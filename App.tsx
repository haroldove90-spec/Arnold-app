import React, { useState, useMemo } from 'react';
import Header from './components/Header';
import MobileNav from './components/MobileNav';
import Dashboard from './components/Dashboard';
import { TOOLS } from './constants';
import type { Tool } from './types';

// Implemented Tools
import ImageGenerator from './components/tools/ImageGenerator';
import ColorPaletteGenerator from './components/tools/ColorPaletteGenerator';
import PlaceholderTool from './components/tools/PlaceholderTool';
import VisualBrainstorming from './components/tools/VisualBrainstorming';
import AutoMockups from './components/tools/AutoMockups';
import AutoTagging from './components/tools/AutoTagging';
import ColorCompatibility from './components/tools/ColorCompatibility';
import BgRemoval from './components/tools/BgRemoval';
import ImageExpansion from './components/tools/ImageExpansion';
import ResolutionUpscaling from './components/tools/ResolutionUpscaling';
import PhotoRestoration from './components/tools/PhotoRestoration';
import TrendAnalysis from './components/tools/TrendAnalysis';
import ABTesting from './components/tools/ABTesting';
import SocialMediaOptimization from './components/tools/SocialMediaOptimization';
import PlagiarismDetection from './components/tools/PlagiarismDetection';
import AutomatedCritique from './components/tools/AutomatedCritique';
import ImprovementSuggestions from './components/tools/ImprovementSuggestions';
import ErrorDetection from './components/tools/ErrorDetection';
import AutoStyleGuides from './components/tools/AutoStyleGuides';
import PresentationScriptGenerator from './components/tools/PresentationScriptGenerator';
import LogoDesign from './components/tools/LogoDesign';
import WebUIDesign from './components/tools/WebUIDesign';
import PackagingDesign from './components/tools/PackagingDesign';
import Typography from './components/tools/Typography';
import AIVectorizer from './components/tools/AIVectorizer';
import ProductPhotography from './components/tools/ProductPhotography';
import BrandNameGenerator from './components/tools/BrandNameGenerator';
import SemanticSearch from './components/tools/SemanticSearch';
import TaskAutomation from './components/tools/TaskAutomation';
import VoiceToDesign from './components/tools/VoiceToDesign';
import BrandIdentityKit from './components/tools/BrandIdentityKit';
import MoodBoardGenerator from './components/tools/MoodBoardGenerator';
import LiveDesignSession from './components/tools/LiveDesignSession';
import VideoStoryboardGenerator from './components/tools/VideoStoryboardGenerator';
import SlideshowGenerator from './components/tools/SlideshowGenerator';
import ConversationalAssistant from './components/tools/ConversationalAssistant';


const App: React.FC = () => {
    const [activeToolId, setActiveToolId] = useState<string | null>(null);

    const handleSelectTool = (tool: Tool) => {
        if (tool.isImplemented) {
            setActiveToolId(tool.id);
        } else {
            // Optionally, show a "coming soon" message
            alert(`The tool "${tool.title}" is coming soon!`);
        }
    };

    const handleBackToDashboard = () => {
        setActiveToolId(null);
    };

    const activeTool = useMemo(() => {
        return TOOLS.find(t => t.id === activeToolId) || null;
    }, [activeToolId]);

    const renderActiveTool = () => {
        if (!activeTool) return null;

        const toolProps = { tool: activeTool, onBack: handleBackToDashboard };

        switch (activeTool.id) {
            // Creative Assistance
            case 'design-from-prompt': return <ImageGenerator {...toolProps} />;
            case 'live-design-session': return <LiveDesignSession {...toolProps} />;
            case 'voice-to-design': return <VoiceToDesign {...toolProps} />;
            case 'smart-color-palettes': return <ColorPaletteGenerator {...toolProps} />;
            case 'visual-brainstorming': return <VisualBrainstorming {...toolProps} />;
            case 'auto-mockups': return <AutoMockups {...toolProps} />;
            case 'brand-name-generator': return <BrandNameGenerator {...toolProps} />;
            case 'mood-board-generator': return <MoodBoardGenerator {...toolProps} />;
            case 'video-storyboard-generator': return <VideoStoryboardGenerator {...toolProps} />;
            
            // Workflow
            case 'auto-tagging': return <AutoTagging {...toolProps} />;
            case 'color-compatibility': return <ColorCompatibility {...toolProps} />;
            case 'semantic-search': return <SemanticSearch {...toolProps} />;
            case 'task-automation': return <TaskAutomation {...toolProps} />;

            // Advanced Editing
            case 'bg-removal': return <BgRemoval {...toolProps} />;
            case 'image-expansion': return <ImageExpansion {...toolProps} />;
            case 'resolution-upscaling': return <ResolutionUpscaling {...toolProps} />;
            case 'photo-restoration': return <PhotoRestoration {...toolProps} />;
            case 'ai-vectorizer': return <AIVectorizer {...toolProps} />;
            case 'product-photography': return <ProductPhotography {...toolProps} />;
            
            // Analysis & Optimization
            case 'trend-analysis': return <TrendAnalysis {...toolProps} />;
            case 'ab-testing': return <ABTesting {...toolProps} />;
            case 'plagiarism-detection': return <PlagiarismDetection {...toolProps} />;
            case 'social-media-optimization': return <SocialMediaOptimization {...toolProps} />;

            // Design Assistant
            case 'conversational-assistant': return <ConversationalAssistant {...toolProps} />;
            case 'automated-critique': return <AutomatedCritique {...toolProps} />;
            case 'improvement-suggestions': return <ImprovementSuggestions {...toolProps} />;
            case 'error-detection': return <ErrorDetection {...toolProps} />;
            case 'auto-style-guides': return <AutoStyleGuides {...toolProps} />;
            case 'presentation-script-generator': return <PresentationScriptGenerator {...toolProps} />;
            case 'slideshow-generator': return <SlideshowGenerator {...toolProps} />;

            // Project Specific
            case 'logo-design': return <LogoDesign {...toolProps} />;
            case 'brand-identity-kit': return <BrandIdentityKit {...toolProps} />;
            case 'web-ui-design': return <WebUIDesign {...toolProps} />;
            case 'packaging-design': return <PackagingDesign {...toolProps} />;
            case 'typography': return <Typography {...toolProps} />;
            
            default:
                return <PlaceholderTool {...toolProps} />;
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
            <Header />
            <main className="flex-grow container mx-auto px-4 py-8 md:py-12 pb-24 md:pb-12">
                {activeTool ? renderActiveTool() : <Dashboard onSelectTool={handleSelectTool} />}
            </main>
            <MobileNav onHomeClick={handleBackToDashboard} />
        </div>
    );
};

export default App;