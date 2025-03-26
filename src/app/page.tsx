import Image from "next/image";
import Assistant from "@/components/assistant";

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-8 lg:p-24">
      <Assistant />
    </main>
  );
}