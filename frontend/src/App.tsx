// src/App.tsx
import { Container, Title } from '@mantine/core';
import QueryForm from './components/QueryForm';
import { MantineProvider } from '@mantine/core';
import '@mantine/core/styles.css';

export default function App() {
  return (
    <MantineProvider>
      <Container size="sm" pt="xl">
        <Title order={1}>Legal Document Retrieval</Title>
        <QueryForm />
      </Container>
    </MantineProvider>
  );
}
