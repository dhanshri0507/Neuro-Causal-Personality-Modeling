// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================
import React from "react";

type Props = {
  value: string;
  onChange: (text: string) => void;
  placeholder?: string;
  rows?: number;
  id?: string;
  style?: React.CSSProperties;
  className?: string;
};

/**
 * TextInput - stateless controlled textarea component.
 *
 * - Controlled via `value` prop.
 * - Calls `onChange(text)` on every keystroke.
 * - No internal state, no validation, no side effects.
 */
export default function TextInput({
  value,
  onChange,
  placeholder,
  rows = 6,
  id,
  style,
  className,
}: Props) {
  return (
    <textarea
      id={id}
      className={className}
      value={value}
      placeholder={placeholder}
      rows={rows}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: "100%",
        padding: "10px",
        fontSize: 14,
        borderRadius: 6,
        border: "1px solid #ccc",
        resize: "vertical",
        fontFamily: "inherit",
        ...style,
      }}
    />
  );
}