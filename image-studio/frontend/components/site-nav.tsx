import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";

export function SiteNav() {
  return (
    <div className="flex items-center justify-between">
      <Link href="/" aria-label="Bonsai home">
        <span
          role="img"
          aria-label="Bonsai"
          className="block h-8 w-[10rem] bg-accent"
          style={{
            WebkitMaskImage: "url('/brand/bonsai-logo-horizontal-dark.svg')",
            maskImage: "url('/brand/bonsai-logo-horizontal-dark.svg')",
            WebkitMaskRepeat: "no-repeat",
            maskRepeat: "no-repeat",
            WebkitMaskSize: "contain",
            maskSize: "contain",
            WebkitMaskPosition: "left center",
            maskPosition: "left center",
          }}
        />
      </Link>

      <ThemeToggle />
    </div>
  );
}
