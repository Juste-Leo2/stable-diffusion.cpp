import * as React from "react";
import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.ComponentProps<"textarea">
>(({ className, ...props }, ref) => {
  return (
    <textarea
      ref={ref}
      className={cn(
        "flex min-h-[160px] w-full resize-none rounded-[1.75rem] border border-border-strong bg-surface-strong px-5 py-4 text-base leading-7 text-foreground outline-none transition placeholder:text-muted focus:border-accent focus:ring-2 focus:ring-accent-ring",
        className,
      )}
      {...props}
    />
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
