import {
  getParameters,
  getPromptLatentActivations,
  getPromptLogitsInput,
  getPromptTokens,
} from "~/lib/api";
import PageComponent from "./page.client";

// Opt out of caching. Otherwise, responses persist when loading different models.
export const fetchCache = "force-no-store";

export default async function Page({ params }: { params: { prompt: string } }) {
  const prompt = decodeURIComponent(params.prompt);

  const parameters = await getParameters();
  const tokens = await getPromptTokens(prompt);
  const latentActivations = await getPromptLatentActivations(prompt);
  const logitsSource = await getPromptLogitsInput(prompt);

  return (
    <PageComponent
      parameters={parameters}
      prompt={prompt}
      tokens={tokens}
      latentActivations={latentActivations}
      logitsSource={logitsSource}
    />
  );
}
