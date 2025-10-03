import type { VercelRequest, VercelResponse } from '@vercel/node';
import { GoogleGenAI, Modality, Type } from "@google/genai";

const getAiInstance = () => {
    const apiKey = process.env.API_KEY;
    if (!apiKey) {
        throw new Error("API_KEY is not configured in Vercel environment variables.");
    }
    return new GoogleGenAI({ apiKey });
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const ai = getAiInstance();
        const { action, payload } = req.body;

        switch (action) {
            case 'generateImage': {
                const { prompt, numberOfImages, aspectRatio } = payload;
                const numImages = typeof numberOfImages === 'number' && numberOfImages > 0 ? numberOfImages : 2;
                const ar = typeof aspectRatio === 'string' ? aspectRatio : '1:1';

                const response = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt,
                    config: { 
                        numberOfImages: numImages, 
                        outputMimeType: 'image/jpeg', 
                        aspectRatio: ar 
                    },
                });
                const imageUrls = response.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
                return res.status(200).json({ data: imageUrls });
            }

            case 'generateStoryboard': {
                const { sceneDescription } = payload;
                const prompt = `Generate a 4-panel cinematic storyboard for the following scene: "${sceneDescription}".
                Each image must represent a key moment in sequence:
                1. A wide shot establishing the scene.
                2. A medium shot focusing on the main action or character.
                3. A close-up on a key detail or expression.
                4. A concluding shot that resolves the action.
                The style should be consistent across all panels.`;
                
                const response = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt,
                    config: { 
                        numberOfImages: 4, 
                        outputMimeType: 'image/jpeg', 
                        aspectRatio: '16:9' 
                    },
                });
                const imageUrls = response.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
                return res.status(200).json({ data: imageUrls });
            }

            case 'generateStructuredText': {
                 const { prompt, responseSchema } = payload;
                 const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: prompt,
                    config: { responseMimeType: "application/json", responseSchema },
                 });
                 return res.status(200).json({ data: JSON.parse(response.text.trim()) });
            }

            case 'generateText': {
                const { prompt } = payload;
                const response = await ai.models.generateContent({ model: "gemini-2.5-flash", contents: prompt });
                return res.status(200).json({ data: response.text });
            }

            case 'generateTextWithImage': {
                const { prompt, image, isJson } = payload;
                const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
                const textPart = { text: prompt };
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: { parts: [imagePart, textPart] },
                    ...(isJson && { config: { responseMimeType: "application/json" } })
                });
                const textResponse = response.text;
                if (isJson) {
                    try {
                        return res.status(200).json({ data: JSON.parse(textResponse) });
                    } catch (e) {
                        return res.status(200).json({ data: { rawResponse: textResponse } });
                    }
                }
                return res.status(200).json({ data: textResponse });
            }

            case 'editImage': {
                const { prompt, image } = payload;
                const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
                const textPart = { text: prompt };
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: { parts: [imagePart, textPart] },
                    config: { responseModalities: [Modality.IMAGE, Modality.TEXT] },
                });

                let resultText: string | null = null;
                let resultImage: string | null = null;

                for (const part of response.candidates[0].content.parts) {
                    if (part.text) resultText = part.text;
                    else if (part.inlineData) {
                        resultImage = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                    }
                }
                
                if (!resultImage) {
                   throw new Error("The AI did not return an edited image.");
                }
                
                return res.status(200).json({ data: { text: resultText, image: resultImage } });
            }

            case 'analyzeOriginality': {
                 const { prompt, image } = payload;
                 const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
                 const textPart = { text: prompt };

                 // Step 1: Get the text analysis
                 const textResponse = await ai.models.generateContent({
                     model: "gemini-2.5-flash",
                     contents: { parts: [imagePart, textPart] },
                     config: { responseMimeType: "application/json" },
                 });
                 
                 let analysisResult;
                 try {
                    analysisResult = JSON.parse(textResponse.text.trim());
                 } catch(e) {
                    return res.status(200).json({ data: { rawResponse: textResponse.text }});
                 }

                 // Step 2: Generate images based on the analysis
                 let similarImages: string[] = [];
                 if (analysisResult.similarDesigns && analysisResult.similarDesigns.length > 0) {
                     const imagePrompts = analysisResult.similarDesigns.slice(0, 2).map((d: any) => `A clear, high-quality image of: ${d.title} - ${d.description}`);
                     
                     if (imagePrompts.length > 0) {
                        const imageGenResponse = await ai.models.generateImages({
                            model: 'imagen-4.0-generate-001',
                            prompt: imagePrompts[0], 
                            config: { numberOfImages: imagePrompts.length, outputMimeType: 'image/jpeg', aspectRatio: '1:1' },
                        });
                        similarImages = imageGenResponse.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
                     }
                 }
                 
                 const finalData = { ...analysisResult, similarImages };
                 return res.status(200).json({ data: finalData });
            }

            case 'generateBrandKit': {
                const { image } = payload;
                const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };

                const analysisPrompt = `Analyze this logo. Extract its primary and secondary colors (give hex codes). Suggest a suitable headline and body font pairing from Google Fonts that matches its style, and provide a short reason for the pairing. Describe the logo's overall style in 3-4 keywords. Respond in JSON format: { "colors": { "primary": ["#hex1"], "secondary": ["#hex2"] }, "typography": { "headlineFont": "Font Name", "bodyFont": "Font Name", "reason": "Why it works..." }, "styleKeywords": ["keyword1", "keyword2"] }`;
                const analysisResponse = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: { parts: [imagePart, { text: analysisPrompt }] },
                    config: { responseMimeType: "application/json" }
                });

                let analysisResult;
                try {
                    analysisResult = JSON.parse(analysisResponse.text.trim());
                } catch (e) {
                    return res.status(500).json({ error: "Failed to parse analysis from AI.", details: analysisResponse.text });
                }

                const styleDesc = analysisResult.styleKeywords.join(', ');
                const mockupPrompt = `Generate 2 different, professional mockups (e.g., a business card, a sign, a website header) that would feature a logo with this style: ${styleDesc}.`;
                
                const mockupResponse = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt: mockupPrompt,
                    config: { numberOfImages: 2, outputMimeType: 'image/jpeg', aspectRatio: '4:3' },
                });
                
                const mockups = mockupResponse.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);

                const finalData = { ...analysisResult, mockups };
                return res.status(200).json({ data: finalData });
            }

            case 'generateMoodBoard': {
                const { theme } = payload;

                const imageResponse = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt: `A mood board for the theme '${theme}'. Generate visually cohesive images including textures, environments, objects, and typography that capture this mood.`,
                    config: { numberOfImages: 9, outputMimeType: 'image/jpeg', aspectRatio: '1:1' },
                });
                const images = imageResponse.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
                
                if (images.length === 0) {
                    throw new Error("Failed to generate images for the mood board.");
                }

                const firstImageB64 = imageResponse.generatedImages[0].image.imageBytes;
                const colorPrompt = "Analyze this image and extract the 5 most dominant colors. Return them as a JSON array of hex code strings, like [\"#RRGGBB\", \"#RRGGBB\"].";
                const colorImagePart = { inlineData: { data: firstImageB64, mimeType: 'image/jpeg' } };
                
                const colorResponse = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: { parts: [colorImagePart, { text: colorPrompt }] },
                    config: { responseMimeType: "application/json", responseSchema: { type: Type.ARRAY, items: { type: Type.STRING } } }
                });

                let colors: string[] = [];
                try {
                    colors = JSON.parse(colorResponse.text.trim());
                } catch(e) {
                    console.error("Failed to parse colors from AI, using fallback.");
                    colors = ["#cccccc", "#999999", "#666666", "#333333", "#000000"];
                }

                return res.status(200).json({ data: { images, colors } });
            }

            case 'generateSlideshow': {
                const { images, title, points } = payload;
                const imageParts = images.map((image: any) => ({
                    inlineData: { data: image.data, mimeType: image.mimeType }
                }));

                const prompt = `Act as a creative director. Create a JSON structure for a client presentation slideshow.
                - The presentation title is: "${title}".
                - Key project points are: "${points}".
                - There are ${images.length} images provided, which you can refer to by their index (0 to ${images.length - 1}).
                - Structure the slideshow logically. Start with a title slide, then use the images to explain the project, summarize the key points, and end with a thank you slide.
                - For slides that include an image, write a concise, professional text that explains what the image represents in the context of the project.
                - Use a variety of slide types.
                - Be smart about which image goes on which slide. For example, a logo image should go on a slide explaining the logo.
                `;

                const schema = {
                    type: Type.OBJECT,
                    properties: {
                        slides: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    type: { type: Type.STRING, enum: ['title', 'image_left', 'image_right', 'full_image', 'bullet_points', 'end'] },
                                    title: { type: Type.STRING },
                                    subtitle: { type: Type.STRING },
                                    text: { type: Type.STRING },
                                    image_index: { type: Type.INTEGER },
                                    caption: { type: Type.STRING },
                                    points: { type: Type.ARRAY, items: { type: Type.STRING } }
                                }
                            }
                        }
                    },
                    required: ["slides"]
                };
                
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: { parts: [...imageParts, { text: prompt }] },
                    config: { responseMimeType: "application/json", responseSchema: schema }
                });

                return res.status(200).json({ data: JSON.parse(response.text.trim()) });
            }

            case 'continueConversation': {
                const { history, newMessage } = payload;
                const contents = history.map((msg: { role: string; text: string }) => ({
                    role: msg.role,
                    parts: [{ text: msg.text }],
                }));
                contents.push({ role: 'user', parts: [{ text: newMessage }] });

                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents,
                    config: {
                        systemInstruction: "You are a world-class senior graphic designer and creative director, acting as a helpful AI assistant for a designer named Arnold. Your name is 'Arnold's Assistant'. Provide expert advice, critiques, and creative ideas. Be concise, professional, and encouraging. You can also help with technical aspects like suggesting CSS code snippets if asked.",
                    },
                });

                return res.status(200).json({ data: { text: response.text } });
            }

            default:
                return res.status(400).json({ error: 'Invalid action' });
        }
    } catch (error: unknown) {
        console.error(`Error in API handler:`, error);
        const errorMessage = error instanceof Error ? error.message : 'An unknown server error occurred.';
        return res.status(500).json({ error: errorMessage });
    }
}