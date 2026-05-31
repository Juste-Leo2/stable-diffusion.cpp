import type { Metadata } from "next";
import { IBM_Plex_Mono, Rethink_Sans } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";

const rethinkSans = Rethink_Sans({
  variable: "--font-rethink-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Bonsai",
  description: "Bonsai — an on-device image-generation studio.",
  icons: {
    icon: "/brand/bonsai-icon-horizontal-dark.svg",
  },
  openGraph: {
    title: "Bonsai",
    description: "Bonsai — an on-device image-generation studio.",
  },
  twitter: {
    card: "summary",
    title: "Bonsai",
    description: "Bonsai — an on-device image-generation studio.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${rethinkSans.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full text-foreground">
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
