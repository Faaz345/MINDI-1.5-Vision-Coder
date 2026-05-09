import type { User } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

export type CloudProfile = {
  id: string;
  email: string | null;
  first_name: string | null;
  last_name: string | null;
  full_name: string | null;
  phone: string | null;
  avatar_url: string | null;
  plan: string | null;
  appearance_settings: Record<string, unknown> | null;
  updated_at: string | null;
};

export type CloudProject = {
  id: string;
  owner_id: string;
  name: string;
  files: Record<string, string>;
  design_settings: Record<string, unknown>;
  source_prompt: string | null;
  status: string | null;
  last_opened_at: string | null;
  updated_at: string | null;
  created_at: string | null;
};

type SaveProjectInput = {
  id?: string | null;
  ownerId: string;
  name: string;
  files: Record<string, string>;
  designSettings: Record<string, unknown>;
  sourcePrompt?: string;
};

function metadataName(user: User) {
  return (
    user.user_metadata?.full_name ||
    user.user_metadata?.name ||
    [user.user_metadata?.first_name, user.user_metadata?.last_name].filter(Boolean).join(" ").trim() ||
    user.email?.split("@")[0] ||
    "Account"
  );
}

export function profileFromUser(user: User | null): CloudProfile | null {
  if (!user) return null;

  return {
    id: user.id,
    email: user.email ?? null,
    first_name: user.user_metadata?.first_name ?? null,
    last_name: user.user_metadata?.last_name ?? null,
    full_name: metadataName(user),
    phone: user.user_metadata?.phone ?? null,
    avatar_url: user.user_metadata?.avatar_url ?? user.user_metadata?.picture ?? null,
    plan: "free",
    appearance_settings: null,
    updated_at: null,
  };
}

export async function fetchCurrentProfile(user: User | null) {
  if (!supabase || !user) return profileFromUser(user);

  const { data, error } = await supabase
    .from("profiles")
    .select("id,email,first_name,last_name,full_name,phone,avatar_url,plan,appearance_settings,updated_at")
    .eq("id", user.id)
    .maybeSingle();

  if (error) {
    const legacyProfile = await supabase
      .from("profiles")
      .select("id,email,first_name,last_name,full_name,phone,avatar_url,plan,updated_at")
      .eq("id", user.id)
      .maybeSingle();

    if (!legacyProfile.error && legacyProfile.data) {
      return { ...legacyProfile.data, appearance_settings: null };
    }

    console.warn("Unable to load cloud profile", error.message);
    return profileFromUser(user);
  }

  return data ?? profileFromUser(user);
}

export async function saveProfileAppearanceSettings(userId: string, appearanceSettings: Record<string, unknown>) {
  if (!supabase) return false;

  const { error } = await supabase
    .from("profiles")
    .update({ appearance_settings: appearanceSettings })
    .eq("id", userId);

  if (error) {
    console.warn("Unable to save appearance settings", error.message);
    return false;
  }

  return true;
}

export async function listUserProjects(userId: string) {
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("projects")
    .select("id,owner_id,name,files,design_settings,source_prompt,status,last_opened_at,updated_at,created_at")
    .eq("owner_id", userId)
    .is("archived_at", null)
    .order("last_opened_at", { ascending: false })
    .limit(12);

  if (error) {
    console.warn("Unable to load cloud projects", error.message);
    return [];
  }

  return (data ?? []) as CloudProject[];
}

export async function saveProjectToCloud(input: SaveProjectInput) {
  if (!supabase) return null;

  const payload = {
    owner_id: input.ownerId,
    name: input.name,
    files: input.files,
    design_settings: input.designSettings,
    source_prompt: input.sourcePrompt ?? null,
    status: "active",
    last_opened_at: new Date().toISOString(),
  };

  const query = input.id
    ? supabase.from("projects").update(payload).eq("id", input.id).select("id,owner_id,name,files,design_settings,source_prompt,status,last_opened_at,updated_at,created_at").single()
    : supabase.from("projects").insert(payload).select("id,owner_id,name,files,design_settings,source_prompt,status,last_opened_at,updated_at,created_at").single();

  const { data, error } = await query;
  if (error) {
    console.warn("Unable to save cloud project", error.message);
    return null;
  }

  return data as CloudProject;
}

export async function renameCloudProject(projectId: string, name: string) {
  if (!supabase) return null;

  const { data, error } = await supabase
    .from("projects")
    .update({ name })
    .eq("id", projectId)
    .select("id,owner_id,name,files,design_settings,source_prompt,status,last_opened_at,updated_at,created_at")
    .single();

  if (error) {
    console.warn("Unable to rename cloud project", error.message);
    return null;
  }

  return data as CloudProject;
}

export async function touchCloudProject(projectId: string) {
  if (!supabase) return;

  const { error } = await supabase
    .from("projects")
    .update({ last_opened_at: new Date().toISOString() })
    .eq("id", projectId);

  if (error) {
    console.warn("Unable to update project activity", error.message);
  }
}
