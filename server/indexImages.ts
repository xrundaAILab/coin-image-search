/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable dot-notation */
import * as dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import type { PineconeRecord } from "@pinecone-database/pinecone";
import { embedder } from "./embeddings.ts";
import { getEnv, listFiles } from "./utils/util.ts";
import { chunkedUpsert } from "./utils/chunkedUpsert.ts";

dotenv.config();

// Index setup
const indexName = getEnv("PINECONE_INDEX");
const pinecone = new Pinecone({
  apiKey: getEnv("PINECONE_API_KEY"),
  environment: getEnv("PINECONE_ENVIRONMENT"),
});

function* chunkArray<T>(array: T[], chunkSize: number): Generator<T[]> {
  for (let i = 0; i < array.length; i += chunkSize) {
    yield array.slice(i, i + chunkSize);
  }
}

async function embedAndUpsert({
  imagePaths,
  chunkSize,
}: {
  imagePaths: string[];
  chunkSize: number;
}) {
  // Chunk the image paths into batches of size chunkSize
  const chunkGenerator = chunkArray(imagePaths, chunkSize);

  // Get the index
  const index = pinecone.index(indexName);
  console.log(index);

  // Embed each batch and upsert the embeddings into the index
  for await (const imagePaths of chunkGenerator) {
    await embedder.embedBatch(
      imagePaths,
      chunkSize,
      async (embeddings: PineconeRecord[]) => {
        await chunkedUpsert(index, embeddings);
      }
    );
  }
}

const indexImages = async () => {
  try {
    // Create the index if it doesn't already exist
    const indexList = (await pinecone.listIndexes()).map((index) => index.name);
    if (!indexList.includes(indexName)) {
      await pinecone.createIndex({
        name: indexName,
        dimension: 512,
        waitUntilReady: true,
      });
    }
    await embedder.init("Xenova/clip-vit-base-patch32");
    const imagePaths = await listFiles("./data");
    await embedAndUpsert({ imagePaths, chunkSize: 100 });
    return;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export { indexImages };
