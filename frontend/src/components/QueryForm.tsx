// src/components/QueryForm.tsx
import { useState } from 'react';
import { TextInput, Button, Box, Text, LoadingOverlay, Select } from '@mantine/core';
import axios from 'axios';

const modelOptions = [
  { value: 'tf-idf_retrival', label: 'TF-IDF Retrieval' },
  { value: 'tf_idf_bm25', label: 'TF-IDF + BM25' },
  { value: 'tf_idf_w2v', label: 'TF-IDF + Word2Vec' },
  { value: 'phoBert_sentences-transformers', label: 'PhoBERT + Sentence Transformers' },
  { value: 'tf_idf_glove', label: 'TF-IDF + Glove' }
];

export default function QueryForm() {
  const [query, setQuery] = useState('');
  const [model, setModel] = useState<string | null>(null);
  const [lawId, setLawId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!model || !query) {
      setLawId("Please enter a query and select a model.");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/predict', {
        query,
        model,
      });
      setLawId(res.data.law_id);
    } catch (err) {
      setLawId('Error: Could not retrieve law ID');
    }
    setLoading(false);
  };

  return (
    <Box pos="relative">
      <LoadingOverlay visible={loading} overlayProps={{ radius: "sm", blur: 2 }} />

      <TextInput
        label="Enter your legal query"
        placeholder="ví dụ: Đập phá biển báo “khu vực biên giới” bị phạt thế nào?"
        value={query}
        onChange={(e) => setQuery(e.currentTarget.value)}
        mb="sm"
      />

      <Select
        label="Select Retrieval Model"
        placeholder="Pick a model"
        data={modelOptions}
        value={model}
        onChange={setModel}
        mb="md"
      />

      <Button onClick={handleSubmit}>Search</Button>

      {lawId && <Text mt="md">🧾 Law ID: {lawId}</Text>}
    </Box>
  );
}
