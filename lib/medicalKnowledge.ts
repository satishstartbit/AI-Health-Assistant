// Static medical knowledge documents.
// In production this would be replaced by a proper vector DB (e.g. MongoDB Atlas Vector Search).
// Each document has keywords for fast lookup and content for context injection.

type MedicalDoc = {
  keywords: string[];
  content: string;
};

const MEDICAL_DOCS: MedicalDoc[] = [
  {
    keywords: ["headache", "migraine", "head pain", "head"],
    content:
      "Headaches can be tension-type (stress, poor posture), migraines (throbbing, light sensitivity, nausea), cluster headaches, or secondary to dehydration, eye strain, or sinus congestion. Red flags: sudden 'thunderclap' headache, headache with fever/stiff neck, or neurological symptoms.",
  },
  {
    keywords: ["fever", "temperature", "chills", "hot"],
    content:
      "Fever (>38°C / 100.4°F) is a sign of immune response to infection — viral (cold, flu, COVID) or bacterial. Mild fever: rest, hydration, paracetamol. High fever (>39.4°C / 103°F) in adults or any fever in infants requires prompt medical evaluation.",
  },
  {
    keywords: ["cough", "throat", "sore throat", "cold", "flu"],
    content:
      "Sore throat is commonly caused by viral pharyngitis, strep throat (bacterial), or allergies. Cough may indicate upper respiratory infection, asthma, or post-nasal drip. Strep throat requires antibiotic treatment; viral causes resolve with supportive care.",
  },
  {
    keywords: ["chest", "chest pain", "heart", "palpitation", "shortness of breath", "breathing"],
    content:
      "Chest pain is a serious symptom. Cardiac causes (heart attack, angina) present with pressure/tightness, jaw/arm radiation, and sweating. Pulmonary causes: pneumonia, pulmonary embolism. Musculoskeletal: costochondritis. ANY chest pain with breathing difficulty should be treated as an emergency.",
  },
  {
    keywords: ["stomach", "abdomen", "nausea", "vomiting", "diarrhea", "cramps", "belly"],
    content:
      "Abdominal symptoms may indicate gastroenteritis (viral/bacterial), food poisoning, IBS, appendicitis, or GERD. Vomiting + diarrhea: maintain hydration (ORS). Severe right lower quadrant pain may indicate appendicitis — urgent evaluation needed.",
  },
  {
    keywords: ["dizziness", "dizzy", "vertigo", "lightheaded", "faint", "fainting"],
    content:
      "Dizziness causes include dehydration, orthostatic hypotension, BPPV (vertigo), inner ear infections, anemia, or low blood sugar. Sudden dizziness with neurological symptoms (facial droop, arm weakness, speech difficulty) — call emergency services immediately (stroke signs).",
  },
  {
    keywords: ["fatigue", "tired", "weakness", "exhaustion", "lethargy"],
    content:
      "Fatigue can result from poor sleep, anemia, hypothyroidism, diabetes, depression, or infection. Persistent unexplained fatigue lasting more than 2 weeks warrants blood tests. Ensure adequate sleep (7-9 hrs), hydration, and balanced nutrition.",
  },
  {
    keywords: ["rash", "skin", "itching", "hives", "allergy", "eczema"],
    content:
      "Skin rashes may be allergic reactions (hives, contact dermatitis), eczema, viral infections (chickenpox, measles), or drug reactions. Hives with throat swelling or breathing difficulty = anaphylaxis — emergency. Most mild rashes respond to antihistamines and moisturizers.",
  },
  {
    keywords: ["back pain", "back", "spine", "lower back"],
    content:
      "Back pain is commonly muscular (strain, poor posture) or disc-related. Red flags: pain with bowel/bladder incontinence (cauda equina), unexplained weight loss, fever, or trauma. Most acute back pain resolves with rest, ice/heat, and gentle movement within 6 weeks.",
  },
  {
    keywords: ["anxiety", "stress", "panic", "mental", "depression", "mood"],
    content:
      "Anxiety and stress can manifest physically (palpitations, chest tightness, shortness of breath, sweating). Differentiate from cardiac causes. Panic attacks are intense but not dangerous. Persistent anxiety or depression requires professional mental health support.",
  },
];

const FALLBACK_CONTEXT =
  "General health guidance: Rest adequately, stay hydrated (8+ glasses/day), eat balanced meals, and avoid self-medicating. Monitor symptoms and seek medical attention if they worsen or persist beyond 48–72 hours.";

export function hasMedicalMatch(symptoms: string): boolean {
  const lower = symptoms.toLowerCase();
  return MEDICAL_DOCS.some((doc) => doc.keywords.some((kw) => lower.includes(kw)));
}

export function getMedicalContext(symptoms: string): string {
  const lower = symptoms.toLowerCase();

  const matches = MEDICAL_DOCS.filter((doc) =>
    doc.keywords.some((kw) => lower.includes(kw))
  );

  if (matches.length === 0) return FALLBACK_CONTEXT;

  return matches.map((m) => m.content).join("\n\n");
}
