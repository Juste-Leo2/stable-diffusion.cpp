import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 rounded-2xl text-sm font-semibold tracking-[0.01em] transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 active:translate-y-0 active:duration-75",
  {
    variants: {
      variant: {
        default:
          "bg-cta-bg px-5 py-3 text-cta-ink shadow-[0_1px_0_rgba(255,255,255,0.38)_inset,0_22px_48px_-26px_var(--accent-ring),0_10px_18px_-14px_rgba(0,0,0,0.45)] hover:-translate-y-0.5 hover:bg-accent-strong hover:shadow-[0_1px_0_rgba(255,255,255,0.44)_inset,0_30px_60px_-28px_var(--accent-ring),0_18px_24px_-18px_rgba(0,0,0,0.55)] active:scale-[0.985] active:shadow-[0_1px_0_rgba(255,255,255,0.32)_inset,0_14px_28px_-20px_var(--accent-ring)]",
        outline:
          "border border-border-strong bg-surface-raised/85 px-4 py-2.5 text-foreground shadow-[0_1px_0_0_var(--border)_inset] backdrop-blur-md hover:-translate-y-px hover:border-accent/70 hover:bg-surface-strong/90 active:translate-y-0",
        secondary:
          "bg-accent-soft px-4 py-2.5 text-foreground ring-1 ring-inset ring-border-strong hover:bg-accent-soft/80 hover:ring-accent/50",
        ghost:
          "px-3 py-2 text-muted hover:bg-surface-raised/85 hover:text-foreground active:bg-surface-strong",
      },
      size: {
        default: "h-11",
        sm: "h-9 px-3 text-xs",
        lg: "h-14 px-7 text-base",
        icon: "h-10 w-10 p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size }), className)}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
