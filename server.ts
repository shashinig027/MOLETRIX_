import express from 'express';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json({ limit: '10mb' }));

// Safely retrieve GoogleGenAI client only when key is provided
function getAiClient(): GoogleGenAI | null {
  const key = process.env.GEMINI_API_KEY;
  if (!key || key.trim() === '' || key === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey: key,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Moletrix Molecular Intelligence Engine',
    hasApiKey: !!process.env.GEMINI_API_KEY,
  });
});

// Predict / Analyze Protein with Gemini 3.8 Flash
app.post('/api/predict-protein', async (req, res) => {
  try {
    const { name, sequence, query } = req.body;

    const ai = getAiClient();
    if (!ai) {
      return res.status(200).json({
        fallback: true,
        message: 'No API key configured; using deterministic biophysical prediction engine.',
      });
    }

    const prompt = `You are a world-class structural biologist and molecular intelligence AI for the Moletrix platform.
Analyze the following protein query/sequence:
Protein Name/Target: ${name || query || 'Unknown Target'}
Amino Acid Sequence: ${sequence ? sequence.slice(0, 500) : 'Derive standard sequence if query is a known biological protein'}

Return ONLY a valid JSON object with the following schema:
{
  "name": "Standard scientific name of protein",
  "organism": "Source organism (e.g., Homo sapiens)",
  "uniprotId": "UniProt accession ID if known",
  "standardSequence": "Standard single-letter uppercase 20-amino-acid sequence (minimum 25-150 AA) if query is a name, or clean sequence if provided",
  "description": "2-3 sentence rigorous summary of physiological function and molecular architecture",
  "therapeuticCategory": "e.g. Oncology, Immunology, Metabolic, Biosensor",
  "keyActiveSite": "Primary catalytic or binding site description",
  "aiInsights": {
    "foldingMechanism": "Detailed biochemical explanation of tertiary folding, hydrophobic core packing, and secondary motifs",
    "functionalDomains": ["List of 2-4 key structural/functional domains with residue spans"],
    "therapeuticPotential": "Clinical drug-targeting opportunities, antibody epitopes, or inhibitor design",
    "engineeringRecommendations": [
      "Point mutation 1 with rationale for stability or affinity",
      "Point mutation 2 with rationale for solubility or shelf-life",
      "Formulation or pegylation engineering strategy"
    ],
    "confidenceAssessment": "Evaluation of structural predictability, RMSD estimates, and disorder regions"
  },
  "pockets": [
    {
      "id": "poc-1",
      "name": "Primary Catalytic / Binding Pocket",
      "residues": [12, 15, 28, 31, 35],
      "druggabilityScore": 0.91,
      "volume": 620,
      "description": "Crucial hydrophobic cleft engaging physiological substrate"
    }
  ]
}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      },
    });

    const text = response.text || '{}';
    const parsed = JSON.parse(text);
    return res.json({ success: true, data: parsed });
  } catch (error: any) {
    console.error('Error in /api/predict-protein:', error);
    return res.status(500).json({
      error: 'Failed to generate AI molecular intelligence',
      details: error.message,
    });
  }
});

// Generate Any Molecule from SMILES code or query with Gemini 3.8 Flash
app.post('/api/molecules/generate', async (req, res) => {
  try {
    const { query, focusArea, userName } = req.body;

    const ai = getAiClient();
    if (!ai) {
      return res.status(200).json({
        success: false,
        fallbackNotice: 'No API key configured; searching local repository of 16 verified compounds.',
      });
    }

    const prompt = `You are a world-class computational chemist and molecular stereochemistry engine for Moletrix.
Synthesize complete stereochemical and 3D molecular data for the following chemical query or SMILES code:
Query: "${query}"

Return ONLY a valid JSON object matching this exact schema:
{
  "id": "lowercase_slug",
  "name": "Standard Chemical Name",
  "iupacName": "Full IUPAC Systematic Name",
  "formula": "e.g. C2H6O",
  "smiles": "Valid Canonical SMILES code",
  "inchiKey": "Standard InChIKey",
  "casNumber": "CAS registry number if known",
  "category": "Chemical Classification (e.g. Aliphatic Alcohol, Carboxylic Acid)",
  "chemicalFamily": "acid|base|alcohol|ketone|aldehyde|pharmaceutical|other",
  "description": "2-3 rigorous scientific sentences on chemical behavior, reactivity, and industrial/pharmaceutical applications",
  "properties": {
    "molecularWeight": 46.07,
    "logP": -0.31,
    "tpsa": 20.2,
    "hBondDonors": 1,
    "hBondAcceptors": 1,
    "rotatableBonds": 0,
    "heavyAtomCount": 3,
    "charge": 0,
    "solubility": "Water solubility description",
    "meltingPoint": "e.g. -114.1 °C",
    "boilingPoint": "e.g. 78.37 °C",
    "density": "e.g. 0.789 g/cm³"
  },
  "lipinskiRuleOf5": {
    "passed": true,
    "violations": 0,
    "details": "Evaluation against the 4 Lipinski criteria",
    "mwPass": true,
    "logPPass": true,
    "donorsPass": true,
    "acceptorsPass": true
  },
  "pharmacology": {
    "mechanismOfAction": "Biochemical mechanism",
    "targetReceptors": ["Primary Target 1", "Target 2"],
    "bioavailability": "e.g. 95% rapid absorption",
    "halfLife": "e.g. 2-4 hours",
    "metabolism": "Hepatic CYP enzymes",
    "clearance": "Renal / respiratory excretion"
  },
  "atoms3D": [
    { "id": 0, "element": "C", "symbol": "C", "x": 0.0, "y": 0.0, "z": 0.0 },
    { "id": 1, "element": "C", "symbol": "C", "x": 1.5, "y": 0.0, "z": 0.0 },
    { "id": 2, "element": "O", "symbol": "O", "x": 2.1, "y": 1.2, "z": 0.0 }
  ],
  "bonds3D": [
    { "source": 0, "target": 1, "order": 1 },
    { "source": 1, "target": 2, "order": 1 }
  ]
}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      },
    });

    const text = response.text || '{}';
    const parsed = JSON.parse(text);
    return res.json({ success: true, molecule: parsed });
  } catch (error: any) {
    console.error('Error in /api/molecules/generate:', error);
    return res.status(500).json({
      error: 'Failed to synthesize molecular intelligence',
      details: error.message,
    });
  }
});

// Compare Matrix Multi-Target Synthesis with Gemini 3.8 Flash
app.post('/api/compare-analysis', async (req, res) => {
  try {
    const { targets } = req.body;

    const ai = getAiClient();
    if (!ai) {
      return res.json({
        fallback: true,
        summary: 'Direct biophysical comparative ranking computed using deterministic thermodynamics and stability indices.',
      });
    }

    const prompt = `You are the lead bio-computational chemist for Moletrix.
Synthesize a comparative evaluation matrix across the following candidate proteins:
${JSON.stringify(targets, null, 2)}

Provide a concise, high-level comparative executive summary suitable for inclusion in the Final Report:
1. Candidate Ranking (Best overall therapeutic/biotech candidate and why)
2. Thermostability & Aggregation trade-offs
3. Developability and manufacturability assessment
4. Recommended next engineering step

Return ONLY a valid JSON object:
{
  "recommendedCandidate": "Name of top candidate",
  "rankingRationale": "Detailed justification comparing MW, pLDDT, stability delta G, and solubility",
  "thermostabilityTradeoffs": "Analysis of Tm, aliphatic index, and aggregation risk",
  "developabilityVerdict": "Manufacturability and expression profile assessment",
  "dossierRecommendations": ["Actionable engineering next step 1", "Actionable engineering next step 2"]
}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      },
    });

    const text = response.text || '{}';
    const parsed = JSON.parse(text);
    return res.json({ success: true, analysis: parsed });
  } catch (error: any) {
    console.error('Error in /api/compare-analysis:', error);
    return res.status(500).json({
      error: 'Failed to generate comparative report',
      details: error.message,
    });
  }
});

// In development, hook Vite into express; in production serve static dist
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(port), '0.0.0.0', () => {
    console.log(`Moletrix Molecular Intelligence running on http://0.0.0.0:${port}`);
  });
}

startServer();
